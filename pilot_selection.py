"""Validation-only selection; inputs use fractions, never percentages."""
import random


def validation_subset(rows, seed=42):
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        group = [r for r in rows if r['label'] == label]
        selected.extend(rng.sample(group, 100))
    return sorted(selected, key=lambda r: r['sample_id'])


def select_coefficient(results):
    reference = results['original']['clean_accuracy']
    eligible = []
    for name in ('kl_0p1', 'kl_0p5', 'kl_1'):
        r = results[name]
        if not r['complete']:
            raise ValueError('Unresolved validation attacks: selection incomplete')
        if r['clean_accuracy'] >= reference - .005:
            eligible.append((r['worst_case_accuracy'], r['clean_accuracy'],
                             -r['coefficient'], name))
    return max(eligible)[-1] if eligible else None
