"""Equal-grid mathematical examples, NOT real video observations."""
import evaluator_original as e

labels = [0, 0, 1, 1]
own = [0.1, 0.2, 0.8, 0.9]
donor = [1, 2, 3, 4]
own_ap = e.average_precision(labels, e.rank_percentile(own))
donor_ap = e.average_precision(labels, e.rank_percentile(donor))
assert own_ap == donor_ap == 1.0
print('EXAMPLE 1: shared increasing ranks; AP is perfect but ownership contrast is zero.')
print({'own_AP': own_ap, 'donor_AP': donor_ap, 'TIA_single_pair': own_ap-donor_ap})
wrong = [4, 3, 2, 1]
wrong_ap = e.average_precision(labels, e.rank_percentile(wrong))
assert abs(wrong_ap - 5/12) < 1e-12
print('EXAMPLE 2: reversed donor ranks; own has an advantage on these fixed labels.')
print({'own_AP': own_ap, 'donor_AP': wrong_ap, 'TIA_single_pair': own_ap-wrong_ap})
print('Do not infer understanding or causality. Real study averages 8 donors per target and 134 target effects.')
print('AP is a ranking metric, not the proportion of correct binary decisions. No real results were modified.')
