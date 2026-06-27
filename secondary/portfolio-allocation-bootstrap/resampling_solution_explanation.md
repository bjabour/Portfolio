# Project 3 Solution Explanation

## Overall Goal

This project estimates the allocation to Asset 1 in a two-asset minimum-variance portfolio. The portfolio invests a fraction $\alpha$ in Asset 1 and $1-\alpha$ in Asset 2.

The minimum-variance allocation is:

$$
\alpha
=
\frac{\sigma_Y^2 - \sigma_{XY}}
{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}}
$$

where:

- $\sigma_X^2$ is the variance of Asset 1 returns.
- $\sigma_Y^2$ is the variance of Asset 2 returns.
- $\sigma_{XY}$ is the covariance between Asset 1 and Asset 2 returns.

The solution first estimates $\hat{\alpha}$ using the sample variances and covariance. Then it uses a nonparametric bootstrap to estimate uncertainty around $\hat{\alpha}$ and construct a 95% percentile confidence interval.

## Slide 1: Data

The first slide describes the input dataset and shows the main return patterns before any model or bootstrap is applied.

The data file is `resampling_data.csv`. It contains 100 rows, and each row contains one paired daily return observation:

- `asset1_return`
- `asset2_return`

The word "paired" is important. Each row contains the returns for both assets on the same day, so the two values in a row should stay together during bootstrap sampling. If we resampled each asset separately, we would destroy the covariance structure between the two assets.

### Daily Return Summary

The summary table shows that Asset 1 and Asset 2 have different risk profiles.

Asset 1:

- Mean daily return: $-0.0971\%$
- Standard deviation: $0.8804\%$
- Minimum: $-1.9967\%$
- Median: $-0.1920\%$
- Maximum: $2.4270\%$

Asset 2:

- Mean daily return: $0.0179\%$
- Standard deviation: $1.3880\%$
- Minimum: $-3.6253\%$
- Median: $0.0701\%$
- Maximum: $3.5812\%$

The main point from this table is that Asset 2 is much more volatile than Asset 1. Its standard deviation is about $1.3880\%$, compared with $0.8804\%$ for Asset 1. Asset 2 also has a wider range of observed returns, from $-3.6253\%$ to $3.5812\%$.

This explains why the final minimum-variance portfolio gives a larger weight to Asset 1. Asset 1 has lower daily volatility, so it naturally receives more weight in a variance-minimizing portfolio.

### Variance and Dependence

The variance and dependence table shows:

- Asset 1 variance: $0.00007751$
- Asset 2 variance: $0.00019265$
- Covariance: $-0.00000437$
- Correlation: $-0.0358$

Asset 2's variance is more than twice the variance of Asset 1. This again supports the idea that Asset 2 is the riskier asset.

The covariance is slightly negative, and the correlation is close to zero at $-0.0358$. This means the two assets have almost no linear relationship in this sample. The slight negative covariance is still useful for portfolio construction, because combining assets with low or negative covariance can reduce total portfolio variance.

### Histograms

The two histograms show the marginal return distributions for each asset.

Axes:

- x-axis: daily return for the asset. For example, $-0.02$ means a daily return of about $-2\%$, $0$ means no return, and $0.02$ means about $+2\%$.
- y-axis: count, meaning the number of observed days whose return falls inside each histogram bin.

The Asset 1 histogram is more concentrated near zero, which matches its lower standard deviation. Most of its daily returns fall in a narrower range.

The Asset 2 histogram is more spread out. It has more extreme negative and positive observations, which matches the higher standard deviation and wider min-max range in the summary table.

These histograms make the risk difference visually clear: Asset 2 has larger return swings than Asset 1.

### Paired Return Cloud

The scatter plot shows Asset 1 returns on the x-axis and Asset 2 returns on the y-axis. Each point is one day.

Axes:

- x-axis: Asset 1 daily return for a given day.
- y-axis: Asset 2 daily return for the same day.
- Each point: one paired row from the dataset, so the point represents both asset returns observed on the same date or observation index.

The plot title reports a correlation of about $-0.036$, which confirms the table value. The cloud is fairly round and does not show a strong upward or downward trend. That means the two assets are not strongly correlated in this dataset.

This plot is important because the bootstrap resampling must preserve this paired structure. Each point should be treated as one observation from the joint empirical distribution of the two assets.

## Slide 2: Model, Method and Results

The second slide explains the model formula, the bootstrap method, assumptions, and the final numerical result.

### Model Formula

The portfolio return is:

$$
R_p(\alpha)
=
\alpha X + (1-\alpha)Y
$$

where $X$ is the Asset 1 return and $Y$ is the Asset 2 return.

The goal is to choose $\alpha$ to minimize the portfolio variance. The closed-form minimum-variance solution is:

$$
\alpha
=
\frac{\sigma_Y^2 - \sigma_{XY}}
{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}}
$$

In the code, the unknown population quantities are replaced by sample estimates:

- sample variance of Asset 1
- sample variance of Asset 2
- sample covariance between Asset 1 and Asset 2

That gives the plug-in estimate $\hat{\alpha}$.

### Bootstrap Method

The project requires a nonparametric bootstrap from the empirical distribution function. The empirical distribution gives probability $1/n$ to each observed row.

The code uses $B = 50000$ bootstrap replicates. For each replicate:

1. Generate uniform random numbers $U \sim \operatorname{Uniform}(0,1)$.
2. Convert each uniform number into a row index using:

$$
\text{index} = \lfloor nU \rfloor
$$

3. Use those indices to select paired rows with replacement.
4. Recompute $\alpha^\ast$ on the resampled dataset.
5. Store the bootstrap value $\alpha^\ast$.

This follows the project requirement because it does not use prohibited functions like `np.random.choice`, `random.choice`, `random.sample`, or `sklearn.utils.resample`.

### Assumptions

The assumptions listed on the slide are:

- The 100 daily return pairs are representative of the return process.
- Rows are independent enough for row-level bootstrap inference.
- Paired row resampling preserves covariance between assets.
- No parametric normal return model is required.
- The sample covariance matrix is stable enough for the ratio estimate.

The most important assumption for this project is the row-level bootstrap assumption. Because the portfolio formula depends on both variances and covariance, preserving the pair structure is essential.

### Final Estimate and Confidence Interval

The final estimate is:

$$
\hat{\alpha} = 0.706418
$$

So the estimated minimum-variance portfolio invests:

- $70.6418\%$ in Asset 1
- $29.3582\%$ in Asset 2

The 95% percentile bootstrap confidence interval is:

$$
[0.612544,\ 0.791623]
$$

The confidence interval width is:

$$
0.791623 - 0.612544 = 0.179079
$$

This interval means that, under bootstrap resampling of the observed data, the plausible minimum-variance weight on Asset 1 is roughly between $61.3\%$ and $79.2\%$.

The interval is entirely above $0.5$, so the bootstrap results consistently support placing more than half of the portfolio in Asset 1.

### Percentile CI Calculation

The percentile table shows:

- 2.5% bootstrap quantile: $0.612544$
- 50.0% bootstrap quantile: $0.706524$
- 97.5% bootstrap quantile: $0.791623$

The median bootstrap estimate is very close to the original plug-in estimate, since $0.706524$ is almost the same as $\hat{\alpha}=0.706418$. This suggests that the bootstrap distribution is centered near the original estimate.

## Slide 3: Visualization

The third slide focuses on the most relevant visual evidence for this specific problem: empirical distributions, bootstrap uncertainty, interval stability, portfolio variance, volatility reduction, and covariance preservation.

### Empirical Distribution Functions

The EDF plot shows the empirical cumulative distribution functions for both assets.

Axes:

- x-axis: daily return value. A value such as $-0.02$ means a daily return of about $-2\%$, and $0.02$ means about $+2\%$.
- y-axis: $F_n(\text{return})$, the empirical cumulative probability. It is the fraction of observed returns that are less than or equal to the x-axis value.

For example, if the Asset 1 curve has y-value $0.60$ at x-value $0$, then about $60\%$ of Asset 1's observed daily returns are less than or equal to zero.

The plot is drawn as a step function because there are only 100 observed returns. Each step corresponds to one or more actual observations in the data. The blue curve is Asset 1's EDF, and the red curve is Asset 2's EDF.

This plot is directly connected to the bootstrap method. The nonparametric bootstrap samples from the empirical distribution, not from an assumed normal distribution. The EDF is the distribution that the bootstrap uses.

The Asset 2 EDF stretches farther into both negative and positive returns than the Asset 1 EDF. This is another visual sign that Asset 2 is more volatile.

The EDF also shows why a nonparametric bootstrap is useful here. We do not need to assume the returns are normally distributed. We use the observed distribution directly.

### Bootstrap Distribution of $\alpha^\ast$

The bootstrap histogram shows the sampling distribution of $\alpha^\ast$ across 50000 resamples.

Axes:

- x-axis: bootstrap estimate $\alpha^\ast$, meaning the minimum-variance weight on Asset 1 recomputed from one bootstrap resample.
- y-axis: count, meaning how many bootstrap resamples produced an $\alpha^\ast$ value inside each histogram bin.

The distribution is centered around the original estimate $\hat{\alpha}=0.706$. The green vertical line marks the original plug-in estimate, and the red dashed lines mark the 2.5% and 97.5% bootstrap quantiles.

The histogram is fairly concentrated, which explains why the confidence interval is not extremely wide. Most bootstrap estimates place the optimal Asset 1 weight around 0.65 to 0.77, with fewer estimates far away from the center.

This plot is the main uncertainty visualization in the slide deck.

### Bootstrap CI Stability

The CI stability plot shows how the bootstrap confidence interval changes as more bootstrap replicates are used.

Axes:

- x-axis: number of bootstrap replicates used in the calculation so far.
- y-axis: $\alpha$, the allocation weight to Asset 1.

The purple band shows the evolving 95% confidence interval, the green line shows the bootstrap median, and the gray line shows the original $\hat{\alpha}$.

The important point is that the lines become stable as the number of replicates increases. By the time the full $B=50000$ replicates are used, the confidence interval is no longer moving much.

This supports the reliability of the reported interval. It shows that the final result is not just an artifact of using too few bootstrap samples.

### Sample Portfolio Variance Curve

The variance curve shows the sample portfolio variance for different values of $\alpha$ between 0 and 1.

Axes:

- x-axis: allocation $\alpha$ to Asset 1. At $\alpha=0$, the portfolio is 100% Asset 2. At $\alpha=1$, the portfolio is 100% Asset 1. At $\alpha=0.5$, it is a 50/50 portfolio.
- y-axis: sample variance of the portfolio return $R_p(\alpha)$.

The curve is U-shaped, which is what we expect from a variance function. The vertical green line marks the minimum at $\hat{\alpha}=0.706$. The shaded purple region shows the bootstrap confidence interval for $\alpha$.

This plot connects the final estimate to the actual optimization problem. It shows that the chosen allocation is the point where the estimated portfolio variance is smallest.

The minimum occurs closer to Asset 1 than Asset 2 because Asset 1 has lower variance. But the minimum is not at $\alpha=1$, because including some Asset 2 can still help due to the low/slightly negative covariance.

### Daily Volatility Comparison

The volatility comparison bar chart shows the daily sample standard deviation of:

Axes:

- x-axis: portfolio or asset being compared: Asset 1, Asset 2, 50/50 portfolio, and minimum-variance portfolio.
- y-axis: sample standard deviation of daily returns, displayed as a percentage. This is daily volatility.

- Asset 1: $0.88\%$
- Asset 2: $1.39\%$
- 50/50 portfolio: $0.81\%$
- Minimum-variance portfolio: $0.73\%$

This is one of the most useful practical plots in the deck. It shows that the minimum-variance portfolio has lower volatility than either individual asset and lower volatility than a simple 50/50 portfolio.

The result makes intuitive sense:

- Asset 1 is safer than Asset 2.
- Asset 2 still contributes diversification because correlation is close to zero and slightly negative.
- The optimized combination produces the lowest observed daily volatility.

This plot translates the formula into a practical risk improvement.

### Paired Rows Preserve Covariance

The final scatter plot again shows the paired daily returns, with the covariance and correlation annotated.

Axes:

- x-axis: Asset 1 daily return.
- y-axis: Asset 2 daily return from the same paired row.
- Each point: one observed pair of daily returns.

This plot is included on the visualization slide because it reinforces the most important bootstrap design choice: resample rows, not individual columns.

The covariance is:

$$
\sigma_{XY} \approx -0.000004
$$

The correlation is:

$$
\rho \approx -0.036
$$

Because the correlation is close to zero, the two assets provide diversification. Since the covariance appears directly in the denominator and numerator of the allocation formula, preserving this relationship during bootstrap resampling is necessary for a valid confidence interval.

## Final Interpretation

The solution estimates that the minimum-variance portfolio should allocate about $70.6\%$ to Asset 1 and $29.4\%$ to Asset 2.

The data support this allocation because Asset 1 has much lower volatility than Asset 2. However, the optimal allocation still keeps some weight in Asset 2 because the two assets have almost no correlation, and their slightly negative covariance helps reduce portfolio variance.

The bootstrap confidence interval $[0.612544,\ 0.791623]$ shows uncertainty in the optimal weight, but the interval remains clearly above $0.5$. Therefore, the evidence consistently favors Asset 1 as the larger part of the minimum-variance portfolio.

The final slide confirms the conclusion visually:

- The EDF plot shows the empirical distributions used by the bootstrap.
- The bootstrap histogram shows the uncertainty around $\hat{\alpha}$.
- The CI stability plot shows that 50000 bootstrap replicates are enough for a stable interval.
- The variance curve shows that $\hat{\alpha}$ is the sample variance-minimizing allocation.
- The volatility bar chart shows the practical reduction in daily risk.
- The paired scatter plot shows why row-level resampling is needed to preserve covariance.

Overall, the analysis is consistent: Asset 1 is less risky, Asset 2 adds some diversification, and the optimized portfolio has the lowest observed daily volatility among the compared choices.
