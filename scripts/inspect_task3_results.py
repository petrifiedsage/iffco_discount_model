import pandas as pd

best = pd.read_csv('results/task3_best_tactic_per_product.csv')
full = pd.read_csv('results/task3_tactic_scores_full.csv')

print('BEST TACTIC PER PRODUCT')
print('='*80)
print(f'Total products: {len(best)}')
print(f'Products with score > 0: {(best["score"] > 0).sum()}')
print(f'Products with score = 0: {(best["score"] == 0).sum()}')
print()

# Show zero-score products
zero_products = best[best['score'] == 0][['PRODUCT_GROUP', 'TACTIC_TYPE', 'claim_count']]
print('Products with zero score (no claims):')
print(zero_products.to_string(index=False))
print()

# Show distribution of unknown mappings
print('UNKNOWN MAPPINGS IN FULL SCORES:')
print(f'  Rows with UNKNOWN_PRODUCT: {(full["PRODUCT_GROUP"] == "UNKNOWN_PRODUCT").sum()}')
print(f'  Rows with UNKNOWN_TACTIC: {(full["TACTIC_TYPE"] == "UNKNOWN_TACTIC").sum()}')
print(f'  Rows with both UNKNOWN: {((full["PRODUCT_GROUP"] == "UNKNOWN_PRODUCT") & (full["TACTIC_TYPE"] == "UNKNOWN_TACTIC")).sum()}')
print()

# Show top products again with new data
print('TOP 15 PRODUCTS BY SCORE (after orphan recovery):')
print(best[['PRODUCT_GROUP', 'TACTIC_TYPE', 'score', 'approval_rate', 'claim_count']].head(15).to_string(index=False))
