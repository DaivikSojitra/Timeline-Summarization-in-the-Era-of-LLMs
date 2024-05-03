import argparse
import codecs
import logging

from statistics import mean
from tilse.data import timelines
from tilse.evaluation import rouge

def get_scores(metric_desc, pred_tl, groundtruth, evaluator):
    if metric_desc == "concat":
        return evaluator.evaluate_concat(pred_tl, groundtruth)
    elif metric_desc == "agreement":
        return evaluator.evaluate_agreement(pred_tl, groundtruth)
    elif metric_desc == "align_date_costs":
        return evaluator.evaluate_align_date_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs":
        return evaluator.evaluate_align_date_content_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs_many_to_one":
        return evaluator.evaluate_align_date_content_costs_many_to_one(pred_tl, groundtruth)

stories = ["bpoil", "egypt", "finan", "h1n1", "haiti", "iraq", "libya", "mj", "syria"]
concatr1 = []
concatr2 = []
agreer1 = []
agreer2 = []
alignr1 = []
alignr2 = []

for i in range(len(stories)):
    prepath = "/Evaluation/T17/Predicted/flan-clusters-summary10"+ stories[i] +".txt"
    refpath = "/Evaluation/T17/Gold/outputTimelines"+stories[i] +".txt"

    print("##########################")
    print(stories[i])
    print("##########################")
    
    predicted = timelines.Timeline.from_file(codecs.open(prepath, "r", "utf-8", "replace"))

    temp_ref_tls = []
    temp_ref_tls.append(
        timelines.Timeline.from_file(codecs.open(refpath, "r", "utf-8", "replace"))
    )

    reference_timelines = timelines.GroundTruth(temp_ref_tls)

    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])

    scores = get_scores("concat", predicted, reference_timelines, evaluator)

    concatr1.append(scores["rouge_1"]["f_score"])
    concatr2.append(scores["rouge_2"]["f_score"])

    scores = get_scores("agreement", predicted, reference_timelines, evaluator)

    agreer1.append(scores["rouge_1"]["f_score"])
    agreer2.append(scores["rouge_2"]["f_score"])

    scores = get_scores("align_date_content_costs", predicted, reference_timelines, evaluator)

    alignr1.append(scores["rouge_1"]["f_score"])
    alignr2.append(scores["rouge_2"]["f_score"])


print("concat r1 Score : ", mean(concatr1))
print("concat r2 Score : ", mean(concatr2))
print("agree r1 Score : ", mean(agreer1))
print("agree r2 Score : ", mean(agreer2))
print("align r1 Score : ", mean(alignr1))
print("align r2 Score : ", mean(alignr2))