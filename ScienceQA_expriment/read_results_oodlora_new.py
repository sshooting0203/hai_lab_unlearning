import json, pandas 
from os import path

writer = pandas.ExcelWriter("all_results_scale_new_ood_1_9_2.xlsx", engine='xlsxwriter')
sheet_name = "unlearn"
sheet = writer.book.add_worksheet(sheet_name)

topic_sheet = writer.book.add_worksheet("per_topic")
topic_row = 0
topic_sheet.write_row(topic_row, 0, ["SEED", "LABEL_K", "UNLEARN_D", "EVAL", "topic", "total", "correct", "acc"])
topic_row += 1

row = 0 
def load_acc(json_path: str) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["acc"]
    
def load_per_topic(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f).get("per_topic", {})

UNLEARN_LIST=["biology", "writing-strategies",
              "figurative-language", 
              "units-and-measurement", "physics", 
              "reference-skills", "chemistry", 
              "punctuation", "economics", 
              "grammar", "earth-science",
              ]
# UNLEARN_LIST=["biology", "physics", "chemistry","economics","earth-science"]
# TEST="test_"
TEST=""
SCALE=str(0.1)
#############################################
LABEL_LIST = ["force"]
SEED_LIST = [0]

for LABEL_K in LABEL_LIST:
    for UNLEAN_D in UNLEARN_LIST:
        su_list, du_list, rd_list, cqa_list, oqa_list = [], [], [], [], []

        for SEED in SEED_LIST:
            OUTPUT_DIR = f"./SCALE_{SCALE}_seed_{SEED}_o_unlearn_lora_{LABEL_K}_checkpoints_10_from0" # @ 수정
            rd_path      = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_testRD.json")
            sdtrain_path = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_trainSD.json")
            sdtest_path  = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_testSD.json")
            cqa_path     = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_commonqa.json")
            oqa_path     = path.join(OUTPUT_DIR, f"{TEST}seed{SEED}_{UNLEAN_D}_openbookqa.json")

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

            sheet.write(row, 0, f"SEED: {SEED}")
            sheet.write(row, 1, f"LABEL_K: {LABEL_K}")
            sheet.write(row, 2, f"{UNLEAN_D}")
            sheet.write(row, 3, SU)
            sheet.write(row, 4, DU)
            sheet.write(row, 5, RD)
            sheet.write(row, 6, CQA)
            sheet.write(row, 7, OQA)
            row += 1

            for eval_name, jp in [("trainSD", sdtrain_path), ("testSD", sdtest_path), ("testRD", rd_path), ("CQA", cqa_path), ("OQA", oqa_path)]:
                per_topic = load_per_topic(jp)
                for topic, stats in per_topic.items():
                    topic_sheet.write_row(topic_row, 0, [
                        SEED, LABEL_K, UNLEAN_D, eval_name, topic,
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
        mean_row = [
            "SEED: MEAN(3)",          # 0
            f"LABEL_K: {LABEL_K}",    # 1
            f"UNLEAN_D: {UNLEAN_D}",  # 2
            su_mean,                  # 3
            du_mean,                  # 4
            rd_mean,                  # 5
            cqa_mean,                 # 6
            oqa_mean,                 # 7
            "", "", ""                # 8~10 (총 11칸 맞춤)
        ]
        for col, val in enumerate(mean_row):
            sheet.write(row, col, val)
        row += 2

writer.close()