import torch.nn.functional as F
from torch import nn
import numpy as np
import argparse, torch, os, json, wandb, warnings, pickle, math, copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.modeling_llama_hacked_o import LlamaForCausalLM_ood
from src.ood_utils import set_seed, detection_performance
from sklearn import svm
from sklearn.mixture import GaussianMixture as GMM
from src.ood_data import load
from scipy.stats import norm
from utils.excel import ExcelLogger
import sklearn.covariance
from functools import partial

warnings.filterwarnings("ignore")
torch.set_num_threads(10)

def entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class LlamaForSelector_O3(nn.Module):
    def __init__(
        self,
        llama_causal_ood,          # your LlamaForCausalLM_ood instance
        tokenizer,
        use_layers=(1,2,3),
        proj_dim=256,
        freeze_backbone=True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_layers = tuple(use_layers)

        self.backbone = llama_causal_ood.model
        self.D = llama_causal_ood.config.hidden_size
        self.cat_dim = self.D * len(self.use_layers)
        # self.llm_q = llama_causal_ood
        # self.backbone_q = llama_causal_ood.model
        # # key encoder: deepcopy backbone+proj, but backbone frozen anyway
        # self.llm_k = copy.deepcopy(llama_causal_ood)
        # self.backbone_k = self.llm_k.model
        # for p in self.llm_k.parameters():
        #     p.requires_grad = False

        # projection heads
        # self.proj_q = nn.Sequential(
        #     nn.Linear(self.cat_dim, self.D),
        #     nn.ReLU(),
        #     nn.Linear(self.D, proj_dim),
        # )
        # self.proj_k = copy.deepcopy(self.proj_q)
        # for p in self.proj_k.parameters():
        #     p.requires_grad = False

        # proj layer를 여러개 달고 싶다면 
        # self.proj_layers = nn.ModuleList([
        #     nn.Sequential(nn.Linear(self.D, self.D), nn.ReLU(), nn.Linear(self.D, proj_dim))
        #     for _ in self.use_layers
        # ])

        self.proj = nn.Sequential(
            nn.Linear(self.cat_dim, self.D),
            nn.ReLU(),
            nn.Linear(self.D, proj_dim),
        )
        # MLM head: token-level prediction from early-layer features
        vocab = len(tokenizer)
        self.mlm_head = nn.Sequential(
            nn.Linear(self.cat_dim, self.D),
            nn.GELU(),
            nn.Linear(self.D, vocab),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # self.m = m
        # self.tau = tau

    def forward(self, batch_mlm=None, batch=None, steps=0, dataloader=None):
        with torch.no_grad():
            out_q, _ = self.backbone(
                input_ids=batch_mlm["input_ids"],
                attention_mask=batch_mlm["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            out_k, _ = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hs_q = out_q.hidden_states # 32 x torch.Size([8, 512, 4096])
        hs_k = out_k.hidden_states

        feats_tok = torch.cat([hs_q[li] for li in self.use_layers], dim=-1)  # [B,T,cat_dim] : torch.Size([8, 512, 12288])
        logits_vocab = self.mlm_head(feats_tok)  # [B,T,V] : torch.Size([8, 512, 32001])
        mlm_loss = F.cross_entropy(
            logits_vocab.reshape(-1, logits_vocab.size(-1)).float(),
            batch_mlm["labels"].reshape(-1).long(),
            ignore_index=-100
        )

        # info_loss = 0.0
        # for j, li in enumerate(self.use_layers):
        #     z_1 = torch.mean(hs_q[li], dim=1)  # [B,D]
        #     z_2 = torch.mean(hs_k[li], dim=1)  # [B,D]
        #     z_1 = self.proj[j](z_1)     # [B,P]
        #     z_2 = self.proj[j](z_2).detach()  # [B,P] stop-grad
        #     sim_mat = torch.einsum("nc,ck->nk", [z_1, z_2.T])  # [B,B]
        #     s_dist = F.softmax(sim_mat, dim=1)
        #     info_loss = info_loss + torch.mean(entropy(s_dist))

        sent_q = torch.cat([hs_q[li].mean(dim=1) for li in self.use_layers], dim=-1)  # [B,cat_dim] : torch.Size([8, 12288])
        sent_k = torch.cat([hs_k[li].mean(dim=1) for li in self.use_layers], dim=-1)  # [B,cat_dim] : torch.Size([8, 12288])

        z_q = self.proj(sent_q)                 # [B,P] (grad O)
        z_k = self.proj(sent_k).detach()        # [B,P] (stop-grad)

        sim_mat = z_q @ z_k.t()                 # [B,B] : torch.Size([8, 256])
        s_dist = F.softmax(sim_mat, dim=1)
        info_loss = entropy(s_dist).mean()
        return mlm_loss, info_loss
    
    @torch.no_grad()
    def _forward_backbone(self, input_ids, attention_mask=None, position_ids=None):
        out, _ = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            active_adapters_d=None,
            ood_weight=None,
            lora_start_layer=None,
            olora=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states

    def sample_X_estimator(self, dataloader):
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        # 표본 공분산 Σ을 추정, 그 역행렬도 제공 
        all_layer_features = []
        num_layers = len(self.use_layers)
        for i in range(num_layers):
            all_layer_features.append([])

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            self.eval()
            batch = {key: value.cuda() for key, value in batch.items()}
            hs = self._forward_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch.get("position_ids", None),
            )  # tuple: each [B,T,D]
            for j, li in enumerate(self.use_layers):
                layer_mean_fea = hs[li].mean(dim=1)  # [B,D]
                all_layer_features[j].append(layer_mean_fea.cpu())
            # for i in range(num_layers):
                # layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach() # T차원으로 평균내서 [B,H] feature 만들기

        mean_list, precision_list, fea_list = [], [], []
        for i in range(num_layers):
            all_layer_features[i] = torch.cat(all_layer_features[i], axis=0)
            fea_list.append(F.normalize(all_layer_features[i], dim=-1)) # i번째 레이어의 모든 샘플 feature를 L2 정규화([N,H])

            sample_mean = torch.mean(all_layer_features[i], axis=0)
            X = all_layer_features[i] - sample_mean # 각 샘플에서 평균을 뺀 centered data
            
            group_lasso.fit(X.numpy())
            temp_precision = group_lasso.precision_ # 공분산의 역행렬 
            temp_precision = torch.from_numpy(temp_precision).float()
            
            mean_list.append(sample_mean.cuda())
            precision_list.append(temp_precision.cuda())
        return mean_list, precision_list, fea_list
        # i번째 레이어 pooled feature 평균, i번째 레이어 정밀도 행렬(공분산), i번째 레이어의 모든 샘플 feature를 L2 정규화

    def get_unsup_Mah_score(self, dataloader, sample_mean, precision, fea_list):
        total_mah_scores = []
        num_layers = len(self.use_layers)
        for _ in range(num_layers):
            total_mah_scores.append([])

        self.eval()
        device = next(self.parameters()).device

        # for batch in dataloader:
        for step, batch in enumerate(tqdm(dataloader)):
            batch_all_features = []
            batch = {key: value.cuda() for key, value in batch.items()}

            hs = self._forward_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                position_ids=batch.get("position_ids", None),
            ) 
            batch_all_features = []
            for j, li in enumerate(self.use_layers):
                layer_mean_fea = hs[li].mean(dim=1).detach()  # [B,D]
                batch_all_features.append(layer_mean_fea)
            # all_hidden_feats = outputs.hidden_states
            # for i in range(num_layers):
            #     layer_mean_fea = torch.mean(all_hidden_feats[i], dim=1, keepdim=False).detach()
            #     batch_all_features.append(layer_mean_fea.data)

            for i in range(len(batch_all_features)):
                batch_sample_mean = sample_mean[i]
                out_features = batch_all_features[i]
                zero_f = out_features - batch_sample_mean

                gaussian_score = -0.5 * ((zero_f @ precision[i]) @ zero_f.t()).diag()
                out_feas = F.normalize(out_features, dim=-1)
                cs_score = out_feas @ fea_list[i].t().cuda()

                cs_score = torch.max(cs_score, dim=1)[0]
                all_score = -cs_score * 1000. + gaussian_score
                total_mah_scores[i].extend(all_score.cpu().numpy())

        for j in range(num_layers):
            total_mah_scores[j] = np.expand_dims(np.array(total_mah_scores[j]), axis=1)
        return np.concatenate(total_mah_scores, axis=1)

def train(args, model, tokenizer, train_dataset, test_dataset, benchmarks, save_name):
    log_dir = "./ood_excel_result"
    xlsx_path = os.path.join(log_dir, f"ood_log_{save_name}_{args.unlearn_dataset}_seed{args.seed}.xlsx")
    excel_logger = ExcelLogger(xlsx_path=xlsx_path, sheet_name="log")
    
    def collate_fn_mlm(batch, mlm_collator):
        batch = mlm_collator(batch)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        input_mask = torch.tensor(batch["attention_mask"], dtype=torch.float)
        labels = torch.tensor(batch["labels"], dtype=torch.long)
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
        }
        return outputs

    def collate_fn(batch, data_collator):
        batch = data_collator(batch)
        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        input_mask = torch.tensor(batch["attention_mask"], dtype=torch.float)
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
        }
        return outputs
    
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    train_dataloader_ran = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn_mlm, mlm_collator=mlm_collator),
                                      shuffle=False, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, data_collator=data_collator), 
                                  shuffle=False, drop_last=True)
    
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        { "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay },
        { "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0 },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    acc_g = 0

    def detect_ood(acc_global, seed, ratio=0.8):
        seed = str(seed)
        mean_list, precision_list, fea_list = model.sample_X_estimator(train_dataloader)

        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, data_collator=data_collator))
        # test_mah_vanlia = model.get_unsup_Mah_score(test_dataloader, mean_list, precision_list, fea_list)[:, 1:]
        test_mah_vanlia = model.get_unsup_Mah_score(test_dataloader, mean_list, precision_list, fea_list)
        # train_mah_vanlia = model.get_unsup_Mah_score(train_dataloader, mean_list, precision_list, fea_list)[:, 1:]
        train_mah_vanlia = model.get_unsup_Mah_score(train_dataloader, mean_list, precision_list, fea_list)

        for _, ood_dataset in benchmarks:
            ood_dataloader = DataLoader(ood_dataset, batch_size=args.batch_size, collate_fn=partial(collate_fn, data_collator=data_collator))
            # ood_mah_vanlia = model.get_unsup_Mah_score(ood_dataloader, mean_list, precision_list, fea_list)[:, 1:]
            ood_mah_vanlia = model.get_unsup_Mah_score(ood_dataloader, mean_list, precision_list, fea_list)

            ood_labels = np.ones(shape=(ood_mah_vanlia.shape[0],))
            test_labels = np.zeros(shape=(test_mah_vanlia.shape[0],))

            test_mah_scores = test_mah_vanlia
            ood_mah_scores = ood_mah_vanlia
            train_mah_scores = train_mah_vanlia

            np.random.shuffle(test_mah_scores)
            np.random.shuffle(ood_mah_scores)
            best_ours_AUROC = 0.0

            if args.ood == 'ocsvm':
                c_lr = svm.OneClassSVM(nu=0.1, kernel='linear', degree=2)
                c_lr.fit(train_mah_scores)

                test_scores = c_lr.score_samples(test_mah_scores)
                ood_scores = c_lr.score_samples(ood_mah_scores)
                train_scores = c_lr.score_samples(train_mah_scores)
                
                X_scores = np.concatenate((ood_scores, test_scores))
                Y_test = np.concatenate((ood_labels, test_labels))

                results = detection_performance(X_scores, Y_test, 'mah_logs', tag='TMP')
                neg_resuls = detection_performance(-X_scores, Y_test, 'feats_logs', tag='TMP')

                if sum(results["TMP"].values()) < sum(neg_resuls["TMP"].values()):
                    results = neg_resuls
                best_ours_AUROC = results['TMP']['AUROC']
                print(args.ood + ' current auroc: {:.3f}'.format(best_ours_AUROC))

                gmm_w, x0 = weighting_func_gmm(train_scores, test_scores)
                w_res_in_train = [obtain_weights(x, gmm_w, x0) for x in train_scores]
                w_res_in_test = [obtain_weights(x, gmm_w, x0) for x in test_scores]
                w_res_ood = [obtain_weights(x, gmm_w, x0) for x in ood_scores]

                print(np.mean(np.array(w_res_in_train)), np.mean(np.array(w_res_in_test)), np.mean(np.array(w_res_ood)))

                threshold = np.max(train_scores)  

                test_labels_prediction = (test_scores <= threshold).astype(int)
                ood_labels_prediction = (ood_scores <= threshold).astype(int)
                Y_predict = np.concatenate((ood_labels_prediction, test_labels_prediction))
                acc = (Y_predict == Y_test).mean()
                print('Test set accuracy: {:.3f}'.format(acc))

                excel_logger.log_row({
                    "seed": int(seed),
                    "unlearn_dataset": args.unlearn_dataset,
                    "auroc": float(best_ours_AUROC),

                    "w_mean_train": float(np.mean(w_res_in_train)),
                    "w_mean_test": float(np.mean(w_res_in_test)),
                    "w_mean_ood": float(np.mean(w_res_ood)),

                    "threshold": float(threshold),
                    "x0": float(x0),
                    "acc": float(acc),
                    "best_acc_global": float(acc_global),
                })
                if acc > acc_global:
                    ood_path = f"./ood_checkpoints_{save_name}_{seed}"
                    if not os.path.exists(ood_path):
                        os.mkdir(ood_path)
                    torch.save(mean_list, f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_mean_list_ocsvm.pt")
                    torch.save(precision_list, f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_precision_list_ocsvm.pt")
                    torch.save(fea_list, f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_fea_list_ocsvm.pt")
                    with open(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_gmm_w_ocsvm.pkl", "wb") as output_file:
                        pickle.dump(gmm_w, output_file)
                    with open(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_ocsvm.pkl", "wb") as output_file:
                        pickle.dump(c_lr, output_file)
                    with open(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_threshold_ocsvm.json", 'w') as f:
                        json.dump([x0, threshold, acc], f)
                    print("SAVE", "CURRENT BEST ACC: ", acc)
                    acc_global = acc
                    # model.roberta.save_pretrained(f"{ood_path}/{args.unlearn_dataset}_{args.ood_dataset}_roberta_ocsvm")
                return acc_global

    num_steps = 0
    acc_g = detect_ood(acc_g, args.seed)
    mlm_loss_avg = []
    for epoch in range(int(args.num_train_epochs)):
        print("start training")
        model.zero_grad()
        for batch_ran, batch in zip(tqdm(train_dataloader_ran), tqdm(train_dataloader)):
            model.train()
            batch_ran = {key: value.to(args.device) for key, value in batch_ran.items()}
            batch = {key: value.to(args.device) for key, value in batch.items()}
            outputs = model(batch_ran, batch, num_steps, train_dataloader)
            mlm_loss, moco_loss = outputs
            loss = moco_loss + mlm_loss
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            print('Step:', num_steps, 'mlm_loss: ', mlm_loss.item(), 'moco_loss: ', moco_loss.item())
        acc_g = detect_ood(acc_g, args.seed)
        print("Epoch Accuracy: ", acc_g)

def weighting_func_gmm(train_in_score, test_in_score):
    # 1. fit two gaussians
    mean1, std1 = norm.fit(train_in_score)
    mean2, std2 = norm.fit(test_in_score)
    # 2. build the gaussian mixture model
    gmm = GMM(n_components=2)
    gmm.means_ = np.array([[mean1], [mean2]])
    # gmm.covariances_ = np.array([[[std1**2]], [[std2**2]]])
    gmm.covariances_ = np.array([[[std2 ** 2]], [[std2 ** 2]]])
    gmm.weights_ = np.array([0.5, 0.5])
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
    # center: x0
    # x0 = minimize(lambda x: (gmm_cdf(x, gmm) - 0.5)**2, x0=0).x[0]
    x0 = (mean1 + mean2) / 2
    return gmm, x0

# cumulative prob func
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    cdf_vals = [w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)]
    return np.sum(cdf_vals)

# the cumulative prob for a point
def cumulative_probability(x, gmm):
    return gmm_cdf(x, gmm)

# 6. the cumulative prob for the symmtric point
def symmetric_cumulative_probability(x, x0, gmm):
    symmetric_x = 2 * x0 - x
    return gmm_cdf(symmetric_x, gmm)

def obtain_weights(input_x, gmm, x0):
    cp_x = cumulative_probability(input_x, gmm)
    cp_symmetric_x = symmetric_cumulative_probability(input_x, x0, gmm)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2 # 0.5

    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))
    return w_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="gcyzsl/O3_LLAMA2_ScienceQA", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int) # 256
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=2.0, type=float)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--loss", type=str, default="margin")
    parser.add_argument("--ood", type=str, default="ocsvm") # gmm ocsvm
    parser.add_argument("--unlearn_dataset", default="scienceqa_biology", type=str)
    parser.add_argument("--ood_dataset", type=str, default="ood_scienceqa")
    parser.add_argument("--base_unlearn_path", type=str, default="./data/scienceqa_SD_only_10/scienceqa_biology")
    parser.add_argument("--base_ood_path", type=str, default="./data/scienceqa_RD_only_10/scienceqa_not_biology")
    parser.add_argument("--save_name", type=str, default="proj")
    args = parser.parse_args()

    print('unlearn_dataset', args.unlearn_dataset)
    print('ood_dataset', args.ood_dataset)
    print('base_unlearn_path', args.base_unlearn_path)
    print('base_ood_path', args.base_ood_path)
    # wandb.init(project=args.project_name, name=args.task_name + '-' + str(args.alpha) + "_" + args.loss)
    # wandb.init(mode="disabled")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    llm = LlamaForCausalLM_ood.from_pretrained(
        args.model_name_or_path,
        config=config,
        # torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<mask>"})
        llm.resize_token_embeddings(len(tokenizer))  
    model = LlamaForSelector_O3(
        llama_causal_ood=llm,
        tokenizer=tokenizer,
        use_layers=(1,2,3),
        proj_dim=256,
        freeze_backbone=True,
    )
    model.to(args.device)

    # datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k']
    datasets = [args.unlearn_dataset, args.ood_dataset]
    benchmarks = ()
    for dataset in datasets:
        if dataset == args.unlearn_dataset:
            train_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True, base_unlearn_path=args.base_unlearn_path, base_ood_path=args.base_ood_path)  # biology
        elif dataset == args.ood_dataset:
            _, original_val_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True, base_unlearn_path=args.base_unlearn_path, base_ood_path=args.base_ood_path)  # non biology
            benchmarks = ((dataset, original_val_dataset),) + benchmarks
    train(args, model, tokenizer, train_dataset, test_dataset, benchmarks, save_name=args.save_name)

if __name__ == "__main__":
    main()
