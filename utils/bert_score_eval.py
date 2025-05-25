import argparse
import pandas as pd
from bert_score import score


def collect_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_dataset", type=str, default="sample_based")
    return parser.parse_args()


def get_dataset(path="data/dataset_lyrics.xlsx"):
    df = pd.read_excel(path)
    df = df[["Title", "Lyric", "Age Class tag"]]
    return df


if __name__ == "__main__":
    args = collect_parser()

    age_groups = ["anak", "remaja", "dewasa", "semua usia"]

    # Load synthetic dataset
    if args.synthetic_dataset == "zero_shot":
        synthetic_df = get_dataset("data/generated_lyrics_2.xlsx")
    elif args.synthetic_dataset == "translation_based":
        synthetic_df = get_dataset("data/generated_lyrics.xlsx")
    else:
        synthetic_df = get_dataset("data/synthesized_lyrics.xlsx")

    # Load human dataset
    human_df = get_dataset("data/dataset_lyrics.xlsx")

    # Group by age
    print(f"\n🔍 BERTScore model: Evaluating '{args.synthetic_dataset}' dataset")

    for group in age_groups:
        syn_group = synthetic_df[synthetic_df["Age Class tag"] == group][
            "Lyric"
        ].tolist()
        hum_group = human_df[human_df["Age Class tag"] == group]["Lyric"].tolist()

        if not syn_group or not hum_group:
            print(f"\n⚠️ Skipping age group '{group}' due to insufficient data.")
            continue

        # Expand references
        expanded_candidates = []
        expanded_references = []

        for synth in syn_group:
            expanded_candidates.extend([synth] * len(hum_group))
            expanded_references.extend(hum_group)

        # Run BERTScore
        (P_all, R_all, F1_all), hashname = score(
            expanded_candidates, expanded_references, lang="id", return_hash=True
        )

        # Group scores
        ref_len = len(hum_group)
        grouped_P = [
            P_all[i * ref_len : (i + 1) * ref_len] for i in range(len(syn_group))
        ]
        grouped_R = [
            R_all[i * ref_len : (i + 1) * ref_len] for i in range(len(syn_group))
        ]
        grouped_F = [
            F1_all[i * ref_len : (i + 1) * ref_len] for i in range(len(syn_group))
        ]

        # Aggregate scores
        max_P = [max(p).item() for p in grouped_P]
        max_R = [max(r).item() for r in grouped_R]
        max_F = [max(f).item() for f in grouped_F]

        mean_P = [p.mean().item() for p in grouped_P]
        mean_R = [r.mean().item() for r in grouped_R]
        mean_F = [f.mean().item() for f in grouped_F]

        print(f"\n📂 Age Group: {group}")
        print(f"▶ Model Hash:         {hashname}")
        print(f"▶ Avg Max Precision:  {sum(max_P)/len(max_P):.6f}")
        print(f"▶ Avg Max Recall:     {sum(max_R)/len(max_R):.6f}")
        print(f"▶ Avg Max F1:         {sum(max_F)/len(max_F):.6f}")
        print(f"▶ Avg Mean Precision: {sum(mean_P)/len(mean_P):.6f}")
        print(f"▶ Avg Mean Recall:    {sum(mean_R)/len(mean_R):.6f}")
        print(f"▶ Avg Mean F1:        {sum(mean_F)/len(mean_F):.6f}")
