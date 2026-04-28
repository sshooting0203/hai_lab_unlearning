import json, pandas
from os import path

writer = pandas.ExcelWriter("clinc_with_o3_infer_lora.xlsx", engine="xlsxwriter")
sheet_name = "unlearn"
sheet = writer.book.add_worksheet(sheet_name)

topic_sheet = writer.book.add_worksheet("per_topic")

# ===== 헤더 먼저 작성 =====
sheet.write_row(0, 0, [
    "SEED", "LABEL_K", "UNLEARN_D", "trainSD", "testSD", "testRD", "mrpc", "rte"
])

topic_sheet.write_row(0, 0, [
    "SEED", "LABEL_K", "UNLEARN_D", "EVAL", "topic", "total", "correct", "acc"
])

# ===== 헤더 고정(스크롤해도 맨 위 유지) =====
sheet.freeze_panes(1, 0)
topic_sheet.freeze_panes(1, 0)

# 데이터는 1행부터 시작
row = 1
topic_row = 1

def load_acc(json_path: str) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "acc" in data:
            return data["acc"]
        # prefer contains_acc or exact_acc for new structure
        return data.get("contains_acc", data.get("exact_acc", 0.0))

def load_per_topic(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "per_topic" in data:
            return data["per_topic"]
        
        # Calculate manually for new structure
        results = data.get("results", [])
        per_topic = {}
        for item in results:
            domain = item.get("domain", "unknown")
            is_correct = item.get("exact_correct", False)
            
            if domain not in per_topic:
                per_topic[domain] = {"total": 0, "correct": 0, "acc": 0.0}
            
            per_topic[domain]["total"] += 1
            if is_correct:
                per_topic[domain]["correct"] += 1
        
        for domain in per_topic:
            if per_topic[domain]["total"] > 0:
                per_topic[domain]["acc"] = per_topic[domain]["correct"] / per_topic[domain]["total"] * 100
        
        return per_topic

UNLEARN_LIST = [
    "work",
    "travel",
    "home",
    "banking",
    "credit_cards",
    "kitchen_and_dining",
    "auto_and_commute",
    "utility",
    "small_talk",
]

# TEST = "test_"
TEST=""
utility1 = "mrpc"
utility2 = "rte"
SCALE = str(0.1)

LABEL_LIST = ["force"]
SEED_LIST = [0,1,2]

for LABEL_K in LABEL_LIST:
    for UNLEAN_D in UNLEARN_LIST:
        su_list, du_list, rd_list, cqa_list, oqa_list = [], [], [], [], []

        for SEED in SEED_LIST:
            OUTPUT_DIR = f"./SCALE_{SCALE}_seed_{SEED}_o_unlearn_lora_{LABEL_K}_checkpoints_clinc9_origin"

            rd_path      = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_testRD.json")
            sdtrain_path = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_trainSD.json")
            sdtest_path  = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_testSD.json")
            cqa_path     = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_{utility1}.json")
            oqa_path     = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_{utility2}.json")

            RD  = load_acc(rd_path)
            SU  = load_acc(sdtrain_path)
            DU  = load_acc(sdtest_path)
            CQA = load_acc(cqa_path)
            OQA = load_acc(oqa_path)

            su_list.append(SU)
            du_list.append(DU)
            rd_list.append(RD)
            cqa_list.append(CQA)
            oqa_list.append(OQA)

            sheet.write_row(row, 0, [
                SEED, LABEL_K, UNLEAN_D, SU, DU, RD, CQA, OQA
            ])
            row += 1

            for eval_name, jp in [
                ("trainSD", sdtrain_path),
                ("testSD", sdtest_path),
                ("testRD", rd_path),
                (utility1, cqa_path),
                (utility2, oqa_path)
            ]:
                per_topic = load_per_topic(jp)
                for topic, stats in per_topic.items():
                    topic_sheet.write_row(topic_row, 0, [
                        SEED,
                        LABEL_K,
                        UNLEAN_D,
                        eval_name,
                        topic,
                        stats.get("total", 0),
                        stats.get("correct", 0),
                        stats.get("acc", 0.0),
                    ])
                    topic_row += 1

        su_mean  = sum(su_list)  / len(su_list)
        du_mean  = sum(du_list)  / len(du_list)
        rd_mean  = sum(rd_list)  / len(rd_list)
        cqa_mean = sum(cqa_list) / len(cqa_list)
        oqa_mean = sum(oqa_list) / len(oqa_list)

        sheet.write_row(row, 0, [
            "MEAN", LABEL_K, UNLEAN_D, su_mean, du_mean, rd_mean, cqa_mean, oqa_mean
        ])
        row += 2

writer.close()