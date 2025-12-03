---
title: 'Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms'
date: 2025-12-01
permalink: /posts/2025/12/computational-algebra-with-attention/
tags:
  - Transformer
  - Computational Algebra
  - Border Basis
---

*This post accompanies our paper accepted at **NeurIPS 2025**: [Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms](https://arxiv.org/abs/2512.00054)*

Many problems in cryptography, robotics, and optimization reduce to solving systems of polynomial equations. Unlike linear systems, where Gaussian elimination provides efficient $O(n^3)$ solutions, polynomial systems present considerably greater computational challenges: 
complexity grows exponentially with degree, 
and traditional algorithms often spend a lot of effort on calculations that turn out not to matter in the end.
In this work, we introduce the **Oracle Border Basis Algorithm (OBBA)**, a neural-guided approach that predicts which computations will contribute to the solution. 
Our approach delivers speedups of up to 3.5× over the state-of-the-art, without compromising reliability: every result is rigorously verified, and if any prediction is incorrect, a proven fallback mechanism guarantees correctness.

---

## Polynomial System Solving and Its Challenges

Many computational problems across science and engineering reduce to solving *polynomial* systems—equations such as $x^2 + y^2 = 1$ and $xy = 0.5$. While linear systems admit efficient solutions via Gaussian elimination, polynomial systems are considerably more complex. The computational cost grows exponentially with the degree of the polynomials, and classical algorithms dedicate the majority of their runtime to computations that, in retrospect, contribute nothing to the final solution.

This motivates our central question: can we predict which computations will be useful before performing them? We address this by training a Transformer to serve as an oracle, guiding the classical _Border Basis Algorithm_ to skip redundant work while preserving correctness.

## Background: From Gaussian Elimination to Border Bases

To provide intuition, recall that Gaussian elimination offers a systematic route to **row echelon form** in linear algebra, simplifying linear systems to a point where solutions become directly accessible. The Border Basis Algorithm (BBA) pursues an analogous goal in the polynomial setting, producing a **border basis**—a structured, canonical representation that encodes all solutions to a polynomial system.

A border basis plays a role in polynomial algebra reminiscent of row echelon form in linear algebra. Just as each row in echelon form has a distinct pivot variable, each polynomial in a border basis has a distinct leading monomial such as $x^2$. Together, these polynomials generate an *ideal*—the algebraic structure encoding all solutions to the original system.

Crucially, both structures are **verifiable**: given a candidate output, one can check whether it is valid without re-running the algorithm. This property is what enables our _correctness guarantee_.

One important constraint is that the BBA only applies to systems with *finitely many solutions*, analogous to requiring a linear system to have full rank. For instance, the system $x^2 + y^2 = 1$ and $x = y$ has exactly two solutions, whereas $x^2 + y^2 = 1$ alone has infinitely many (the unit circle) and falls outside the algorithm's scope.

## The Border Basis Algorithm

The core operation of the BBA mirrors Gaussian elimination: maintain a set of basis polynomials and systematically extend it by combining polynomials to eliminate terms. However, unlike linear systems, polynomial systems require working **degree by degree**—starting with low-degree polynomials, then progressively considering higher degrees until the basis stabilizes. The number of potential combinations grows rapidly with each degree, leading to a potentially enormous number of reductions.

## Computational Redundancy in Polynomial Algebra

A linear system can be overdetermined: some equations are linear combinations of others. In Gaussian elimination, this manifests as rows that reduce to zero—redundant equations that, if identified in advance, could simply be omitted.

In the Border Basis Algorithm, an analogous phenomenon occurs at considerably larger scale. Since the algorithm operates degree by degree, at any given iteration we only consider polynomials up to some maximum degree $d$. This defines the current **computational universe** $\mathcal{L}$—the set of all monomials up to degree $d$. For example, with two variables $x$ and $y$ at degree 2, the universe is $\mathcal{L} = \\{1, x, y, x^2, xy, y^2\\}$. The algorithm maintains a **generator set** $\mathcal{V}$ of polynomials with distinct leading terms, tracking only polynomials that lie entirely within $\mathcal{L}$; terms outside this set are deferred to later iterations.

At each iteration, the algorithm proceeds as follows:

1. **Expansion**: Multiply every polynomial in the current basis by every variable, creating a pool of candidate polynomials $\mathcal{V}^+$.
2. **Reduction**: Apply Gaussian elimination to compute a basis for the span of all candidates.
3. **Filtering**: Retain only those polynomials whose terms lie entirely within the computational universe $\mathcal{L}$.

A candidate polynomial **extends the basis** if, after reduction, it produces a non-zero polynomial that was not already expressible as a combination of existing basis elements. If it reduces to zero, it was redundant—merely a consequence of polynomials already present.

In practice, many candidates are redundant: they reduce to zero and contribute nothing new. Reducing these unnecessary candidates is computationally costly and dominates the algorithm's runtime.

### A Worked Example

Consider one iteration at degree 2, where the computational universe is:

$$\mathcal{L} = \{1, x, y, x^2, xy, y^2\}$$

and the current generator set contains two polynomials:

$$\mathcal{V} = \{x - 1,\; x^2 + y^2 - 1\}$$

**Step 1: Expansion.** Multiply each polynomial in $\mathcal{V}$ by each variable ($x$ and $y$) to form $\mathcal{V}^+$:

| Expansion | Result |
|:---------:|:------:|
| $x \cdot (x - 1)$ | $x^2 - x$ |
| $y \cdot (x - 1)$ | $xy - y$ |
| $x \cdot (x^2 + y^2 - 1)$ | $x^3 + xy^2 - x$ |
| $y \cdot (x^2 + y^2 - 1)$ | $x^2y + y^3 - y$ |

**Step 2: Reduction.** Apply Gaussian elimination to compute a basis for the span of $\mathcal{V}^+$:

- $x^2 - x$ reduces (using $x^2 + y^2 - 1$) to $y^2 + x - 1$
- $xy - y$ cannot be further reduced

**Step 3: Filtering.** Retain only polynomials whose terms lie entirely within $\mathcal{L}$. The last two expansions contain $x^3$, $xy^2$, $x^2y$, and $y^3$—monomials outside the computational universe—and are therefore discarded. This yields two new polynomials that extend $\mathcal{V}$:

$$\mathcal{V} \leftarrow \{x - 1,\; x^2 + y^2 - 1,\; y^2 + x - 1,\; xy - y\}$$

Of 4 candidates, only 2 were useful; the remaining 2 fell outside the current scope. This was a minimal example—as problems grow, the ratio of redundant to useful reductions becomes substantially worse. In Gaussian elimination, redundant rows reduce to zero; in border basis computation, the *majority* of generated candidates reduce to zero.

<figure class="fig-white-bg">
  <img src="/images/blogpost_figures/BorderBasisAlgo-1.png" alt="Border Basis Algorithm Visualization">
  <figcaption>Border basis concepts: (a) A border basis with border terms \(\{y^2,xy,x\}\). (b) BBA's iterative expansion of \(\mathcal{V}\), showing leading terms: two initial polynomials yield four expansions, then eight more - though only two out of twelve were necessary. (c) The oracle approach achieves the same result with just four targeted expansions.</figcaption>
</figure>

While the underlying linear algebra phenomenon—linear dependence—is the same, the ratio of redundant to useful candidates is substantially worse in the polynomial setting. The computational universe of possible expansions grows combinatorially, yet the informative directions—new basis elements that survive reduction—are sparse. Despite this, we pay the full cost of generating and reducing every candidate, only to discover that most were unnecessary.

This is precisely the inefficiency our Transformer oracle addresses.

## A Neural Oracle for Expansion Selection

Rather than exhaustively expanding all candidates and discovering afterward that most reduce to zero, we predict in advance which expansions are likely to yield new basis elements.

We train a Transformer to serve as an oracle that examines the current polynomial set and predicts which expansions will produce non-zero results after Gaussian elimination. Instead of computing all expansions and discarding most, we compute only those the oracle recommends.

Recall that the Border Basis Algorithm operates **degree-by-degree**: at each iteration, we consider polynomials within a fixed monomial space, then expand to include higher degrees. This iterative structure provides a natural mechanism for collecting training data—we record which expansions produced non-zero results at each step during standard algorithm execution.

### Maintaining Correctness

A critical feature of our approach is that it **preserves correctness guarantees**. 
Border bases have a natural criteria that makes them typically much easier to verify than compute [1].
If verification fails—indicating that the oracle's predictions were insufficient—we fall back to the standard algorithm. This means:

- If the oracle makes accurate predictions: maximum speedup is achieved
- If the oracle makes errors: verification detects this and triggers recovery
- The algorithm always terminates with a provably correct result

## Technical Contributions

### Efficient Monomial Embedding

Polynomials can contain thousands of terms, leading to extremely long token sequences under standard representations. Consider encoding a polynomial such as $x + 2$ for a Transformer. A naive approach tokenizes each component separately—the coefficient, each variable's exponent, and operators between terms:

```
C1, E1, E0, +, C2, E0, E0
```

This yields 7 tokens for just two terms. For a polynomial with $n$ variables, each term requires $(n+1)$ tokens (one for the coefficient, one exponent per variable), plus operators and separators. This quickly becomes prohibitive for larger problems.

We developed a **monomial-level embedding** that encodes each term as a single token. Rather than decomposing a term such as $3x^2y$ into separate tokens for the coefficient (3), the $x$-exponent (2), and the $y$-exponent (1), we combine this information into one embedding vector:

$$\text{embed}(\text{term}) = \text{embed}_{\text{coef}}(c) + \text{embed}_{\text{exponents}}(a_1, \ldots, a_n) + \text{embed}_{\text{separator}}$$

The exponent embedding aggregates information from all variables:

$$\text{embed}_{\text{exponents}}(a_1, \ldots, a_n) = \frac{1}{n} \sum_{i=1}^n \text{embed}_{\text{var}_i}(a_i)$$

This design aligns with the structure of polynomial algebra, which is fundamentally **term-centric**: when combining polynomials, we match terms with identical monomial structure. By embedding each term as a single token, the Transformer's attention mechanism can directly compare terms across polynomials, rather than first determining which clusters of tokens belong together.

<figure class="fig-white-bg fig-75">
  <img src="/images/blogpost_figures/token_count.png" alt="Token Efficiency">
  <figcaption>The term truncation and monomial embedding significantly reduce input size.</figcaption>
</figure>

This embedding reduces token count by a factor of $\mathcal{O}(n)$ for $n$-variable polynomials, enabling us to handle substantially larger problems within the Transformer's context window.
In addition, we only provide the Transformer with the first $k$ leading terms of the polynomials in $\mathcal{V}$, since those typically determine which terms get eliminated. By combining these strategies, we can dramatically shorten the input sequences.

### Training Data Generation

Generating appropriate training data for polynomial algebra presents its own challenges. Recall that the BBA only applies to systems with finitely many solutions. Random sampling of polynomials almost never produces such systems—most random polynomial combinations have either no solutions or infinitely many.

We address this by **sampling in reverse**: rather than hoping random polynomials happen to possess the required structure, we begin with a known valid border basis (which by definition corresponds to a system with finitely many solutions), then apply random transformations to create diverse training examples while preserving the essential algebraic structure.

## Experimental Results

We evaluate on randomly generated polynomial systems over finite fields—a setting common in cryptographic applications and algebraic coding theory. Each system has a known, finite solution set, enabling automatic verification of correctness.

<figure class="fig-white-bg">
  <img src="/images/blogpost_figures/runtime_barplots-1.png" alt="Runtime Results">
  <figcaption>Runtime comparison (log scale) across different problem configurations. OBBA consistently outperforms the baseline BBA, with speedups increasing for more challenging problems.</figcaption>
</figure>

Of particular interest, the oracle exhibits **strong out-of-distribution generalization**. Models trained exclusively on degree-2 polynomials successfully accelerate computations on degree-3 and degree-4 problems—instances 10–100× harder than anything encountered during training. This suggests the Transformer learns structural properties of useful expansions rather than merely pattern-matching on the training distribution.

<figure class="fig-white-bg">
  <img src="/images/blogpost_figures/ood_speedup-1.png" alt="Out-of-Distribution Performance">
  <figcaption>
    Speedup as problem difficulty increases for systems with 4 variables. OBBA still achieves strong speedups even when solving problems harder than those it was trained on (higher-degree polynomials). The ratio $\frac{|\mathcal{V}|}{|\mathcal{L}|}$ helps us decide how often to use the oracle: if this ratio is close to $1$, we are nearly done, but if it is much smaller, more steps are needed.
  </figcaption>
</figure>

### Numerical Results

For 5-variable polynomial systems over $\mathbb{F}_{32003}$ (a prime field commonly used in computational algebra):

| Degree | Baseline BBA | Our Method | Speedup |
|:------:|:------------:|:----------:|:-------:|
| 2 (in-distribution) | 11.4s | 0.60s | 19× |
| 3 (out-of-distribution) | 40.7s | 1.7s | 24× |
| 4 (out-of-distribution) | 136.7s | 5.6s | 24× |

The out-of-distribution results are notable: problems significantly harder than anything in training, yet the oracle still achieves greater than 20× speedup. When the oracle does make prediction errors, the verification step detects them and triggers fallback—correctness is never compromised.

## Limitations

**Variable count.** We demonstrate results for up to 5 variables. Scaling to higher-dimensional systems will require larger models and more sophisticated training strategies.

**Oracle accuracy on difficult problems.** While out-of-distribution generalization is strong, there are limits. On problems far outside the training distribution, the oracle's precision decreases and more fallbacks are triggered. The algorithm remains correct, but speedups diminish.

## Conclusion and Outlook

This work establishes a connection between neural prediction and classical computational algebra. The Oracle Border Basis Algorithm demonstrates that learned oracles can significantly accelerate symbolic computation while preserving correctness guarantees.

Our primary contributions are:

1. **Guaranteed correctness**: The fallback mechanism ensures the algorithm always produces the correct answer
2. **Efficient representation**: Monomial embeddings reduce token counts by a factor of $O(n)$
3. **Data-efficient training**: Reverse sampling generates diverse, valid training instances
4. **Strong generalization**: Models transfer effectively to harder problems than those seen during training


Longer-term directions include:
- **Larger variable counts** via hierarchical or sparse attention mechanisms
- **Infinite fields** ($\mathbb{Q}$, $\mathbb{R}$) with appropriate numerical handling
- **Integration with computer algebra systems** such as Macaulay2 or Singular

---

**Paper:** [Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms](https://arxiv.org/abs/2512.00054) (NeurIPS 2025)

**Code:** [github.com/HiroshiKERA/OracleBorderBasis](https://github.com/HiroshiKERA/OracleBorderBasis)

```bibtex
@inproceedings{kera2025computational,
      title={Computational Algebra with Attention: Transformer Oracles for Border Basis Algorithms}, 
      author={Hiroshi Kera and Nico Pelleriti and Yuki Ishihara and Max Zimmer and Sebastian Pokutta},
      year={2025},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      eprint={2512.00054},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.00054}, 
}
```

## References 

[1] Kehrein and Kreuzer. Computing border bases. *Journal of Pure and Applied Algebra*, 205(2):279–295, 2006.
