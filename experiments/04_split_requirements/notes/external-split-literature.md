# External literature on homology-aware splits and calibration

Scope: primary sources found by web search, not the local PDFs. Dates are publication or online dates as stated by the source. Claims are labelled **established** (documented practice or a tool's own specification), **proposed** (a paper's method, not a field standard), or **inference** (applied to this benchmark, not stated by the source).

Anything marked "not re-verified" was not opened in this pass. Do not treat those sentences as citations.

## 1. Homology-aware partitioning

**Established / tool paper, not a universal standard.** Graph-Part (Teufel et al., NAR Genomics and Bioinformatics, published online 16 Oct 2023, DOI 10.1093/nargab/lqad088; https://europepmc.org/article/MED/37850036, full text https://pmc.ncbi.nlm.nih.gov/articles/PMC10578201/) builds an all-vs-all similarity graph and partitions it so that no edge crosses the train/test cut. It is a partitioning method, not a remote-homology detector.

What it fixes, relative to CD-HIT-style reduction:

- CD-HIT-style clustering "should not be used for homology partitioning, since the maximum similarity criterion is only applied to the representative sequences" (Teufel et al. 2023, EuropePMC text). Two sequences can both sit under the identity cutoff from the representative and still be close to each other. Splitting on cluster labels then leaks.
- "By utilizing exhaustive global Needleman–Wunsch alignments, GraphPart guarantees separation at its chosen identity threshold. Additionally, other alignment programs such as MMseqs2 can be used by GraphPart" (Teufel et al. 2023). The guarantee is only as good as the edges. MMseqs2 edges do not inherit the Needleman–Wunsch guarantee.
- On some datasets it keeps more data than reduction: "retaining more than 50% more sequences in its partitions on some datasets" (Teufel et al. 2023, paper text via https://www.researchgate.net/publication/374771680). That is a retention claim, not a claim of better remote-homology detection.
- "GraphPart can fail to find a partitioning solution. Homology reduction algorithms then have to be used instead" (Teufel et al. 2023). A failed partition is not a split.

**Inference.** For this project, Graph-Part is useful only after the similarity graph is defined. A 20% pairwise-identity graph will not contain the remote homologs that profile search would have found. Use profile search (or profile-profile, if that is the edge definition) to draw edges, then Graph-Part to cut the graph. Graph-Part does not replace the search.

**Independent sets, different problem.** Petti and Eddy (PLOS Computational Biology, 2022; https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010021) describe sampling so that "split the data such that each test sequence is less than p% identical to any individual training sequence." Only the abstract was opened. This is about sampling within a family for remote-homology detection, not about building a labelled lookup database. Do not cite the full-text bridge-sequence argument or any specific p; those were not re-extracted.

SpanSeq (2024) was not re-opened. No numbers from it.

## 2. Redundancy reduction tools

### MMseqs2

Source for the parameter semantics below: MMseqs2 wiki, read as the user guide (https://github.com/soedinglab/MMseqs2/wiki). No default `--min-seq-id` was verified; do not assume one.

**Established.**

- Default identity is not the fraction of identical columns. Clustering criteria are "by default, defined by a highly correlated measure, the equivalent similarity score of the local alignment (including gap penalties) divided by the maximum of the lengths of the two locally aligned sequence segments." A reported "20%" under the default is not BLAST-style percent identity.
- `--alignment-mode 3` is "the number of identical aligned residues divided by the number of aligned columns including columns containing a gap in either sequence." State which definition the split used.
- Coverage: `--cov-mode 0` is bidirectional (query and target). `--cov-mode 1` is target coverage. `--cov-mode 2` is query coverage. `--cov-mode 3` is named "target-in-query length coverage"; the formula was not extracted, so it is not restated here. `--cov-mode 4` is mapped from CD-HIT `-s`; formula not extracted.
- Clustering modes: `--cluster-mode 0` is greedy set cover (default when cov-mode is 0 or 2). `--cluster-mode 1` is connected component (default when cov-mode is 1). `--cluster-mode 2` is greedy incremental, CD-HIT style, and compares to the representative only. Mode 2 has the same partitioning bug Graph-Part names.
- Cascaded clustering can leave members that no longer meet the criteria because the representative changes. `--cluster-reassign` corrects that. An easy-cluster result is not an all-pairs split.
- Linclust "can cluster sequences down to 50% pairwise sequence similarity and its sensitivity can be adjusted" with `-k`. It is the wrong tool for a 20% split.
- Profile clustering is documented ("How to cluster using profiles"): build a consensus, enrich against UniRef50, search the consensus. The full command block was not extracted. **Established from the guide:** profile search is the documented, more sensitive procedure. Söding 2005 was not opened; do not cite it for sensitivity.
- The prefilter emits pairs, then alignment runs only on those pairs. A low-identity search can miss pairs the prefilter never emits. No false-negative rate was verified.

**Inference.** "MMseqs2 at 20%" is not a specification until identity definition, coverage mode, search type (sequence vs profile vs profile-profile), and whether the comparison was against the full pre-reduction set are all stated.

### CD-HIT and psi-cd-hit

Source checked: CD-HIT user guide (http://weizhongli-lab.org/cd-hit/wiki/doku.php?id=cd-hit_user_guide). Li and Godzik, Bioinformatics, 2006, is the usual paper citation; the guide page is the source that was actually read.

**Established.**

- "CD-HIT does not cluster sequences of low identity. CD-HIT can not cluster proteins at <40% identity."
- psi-cd-hit is the route to 30% or below, and "is much slower."
- Default `-G 1` is global identity: identical residues over the whole sequence. Local identity is a different definition. Mixing the two makes a cutoff incomparable.
- Word length is tied to the identity threshold: 5 for 0.7–1.0, 4 for 0.6–0.7, 3 for 0.5–0.6, 2 for 0.4–0.5.
- Greedy incremental clustering compares each sequence to the representative only. Same failure mode as Graph-Part's critique of representative-only splits.

CD-HIT is appropriate for fast redundancy reduction at or above 40% identity. It is not a 20% homology partitioner.

### Foldseek and structure-based redundancy

van Kempen et al., Nature Biotechnology, online 8 May 2023, journal citation 2024, 42:243–246, DOI 10.1038/s41587-023-01773-0 (https://www.nature.com/articles/s41587-023-01773-0).

**Established, with a documented discrepancy.** Foldseek is the appropriate redundancy tool when structures, or 3Di sequences derived from them, exist. It is not a substitute for a sequence-profile split of sequences that have no structure.

Coverage default is not settled in the sources that were opened. The README parameter table says `-c` default is 0.0. GitHub issue #299 (2024, https://github.com/steineggerlab/foldseek/issues/299) says the help text is wrong and `easy-cluster` actually uses coverage 0.8. Do not assert one default. Set `-c` explicitly and record the binary version.

Do not claim that 3Di links analogous folds; that sentence was not re-checked. Barrio-Hernandez et al., Nature 2023 (AFDB clusters) was not re-opened; no numbers from it.

## 3. Critiques of sequence-identity splits

**Established, from a methods page, not from a new experiment.** AlQuraishi's ProteinNet splitting methodology (https://github.com/aqlaboratory/proteinnet/blob/master/docs/splitting_methodology.md) states the operational failures:

- Separate the test set from the full pre-thinning training set, not from reduced representatives. Similarity is not transitive. Two sequences can both be distant from a representative and close to each other.
- Sequence-sequence identity, even a strict cutoff, does not separate CATH topologies. ProteinNet uses profile-profile search and a threshold of about 10% for topology-level separation. Profile-profile is more sensitive than sequence-sequence or sequence-profile.
- Domain-in-full-chain matches leak if only domain-versus-domain comparisons are run.
- Validation must be built with the same separation logic as the test set.
- The page restates the twilight zone: below 20–30% identity, sequences can still be structurally similar. Rost 1999 was not re-opened. Cite this page for that formulation, not Rost.

The BMC Bioinformatics 2019 ProteinNet paper URL was not re-opened. Cite the methodology page.

**Established, from the database's own rules, not from a benchmark paper.** CATH download page (https://www.cathdb.info/download): sequence-redundant subsets "have no pair of domains (according to BLAST) with >= 20 or 40% sequence identity (depending on the data set chosen), over 60% overlap (over the longer sequence." The construction example shown on the page that was read is the 40% case (S40). The page does not spell out a separate S20 recipe beyond that bullet. Older S35/S95 names were not on the page that was read; do not assert them.

CATH homologous-superfamily criteria (https://www.cathdb.info/wiki/doku.php?id=faq_about_cath): a domain joins a homologous superfamily if it has "significant sequence similarity (sequence identity >= 35%, overlap >= 60%)", or "significant structural similarity (SSAP score >= 80) and some sequence similarity (sequence identity >= 20%)", or SSAP >= 70 plus overlap and function, or an HMM-HMM hit. CATH itself does not treat 20% sequence identity as enough to say two domains are unrelated.

**A leakage number that does not transfer.** Hermann et al., PMLR 2024, "Homology leakage in protein language models" (https://proceedings.mlr.press/v235/hermann24a.html). Abstract only. They report "an increase of 11.1% in the Spearman's rank correlation coefficient for the stability of protein structures (measured by the melting point)" when comparing a leaked pretraining setup (FLIP 0.603) with EPA (0.548). That is melting-point Spearman inflation from pretraining leakage, not CATH nearest-neighbour accuracy inflation at a 20% split. Do not reuse 11.1% for this project's split.

No paper quantifying CATH accuracy inflation at a 20% identity split was opened. Brenner, Chothia and Hubbard, PNAS 1998, was not opened. Do not cite it.

**Inference.** A 20% profile-search filter is a minimum, not proof that structural homologs are gone. For a CATH topology or architecture task, ProteinNet's profile-profile threshold near 10%, plus an explicit coverage rule, is the stricter documented bar. Whether this project can afford profile-profile search is a cost question, not a reason to pretend pairwise 20% does the same job.

## 4. Benchmark design in protein ML

What was actually opened:

- TAPE, Rao et al., NeurIPS 2019, arXiv:1906.08230 (https://arxiv.org/abs/1906.08230). Abstract only: they "hold out entire evolutionary groups." The abstract does not contain an exact fold/superfamily sentence. Do not invent one.
- CAFA2, Jiang et al., Genome Biology, 2016 (https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1037-6). A ResearchGate extract of that paper says "we refer to m(0)/n as the coverage." Partial evaluation counts proteins with at least one prediction above a threshold. **Established:** when a method may abstain, report coverage with the precision-like score.

**Inference, labelled as such.** An nearest-neighbour distance threshold is an abstention rule. Precision at that threshold without coverage describes only the queries that were answered. Remote queries will abstain more often, so a high precision on a tiny answered fraction is not a benchmark result.

PEER, FLIP, ProteinGym, and CARE were not re-opened. Do not attribute identity cutoffs, split rules, or warnings to them. FLIP is often described as mutation and landscape splits rather than a homology-reduction standard; that characterization was not re-verified, so it is not used here. Radivojac et al., Nature Methods 2013 (Fmax) was not re-opened.

**Established, from ProteinNet's methodology page (same URL as section 3).** The validation set must be constructed with the same separation logic as the test set. A calibration set that is closer to the lookup set than the test set is not a validation set for this task.

## 5. Calibration of confidence scores

**Established, general statistics, not biology-specific.**

- Angelopoulos and Bates, "A Gentle Introduction to Conformal Prediction." A readable HTML version states that the coverage guarantee holds for any calibration size and suggests "a moderate number (e.g., 500)" because small calibration sets fluctuate. The PDF at https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf was not the source used, and https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf was not opened as a PDF. Treat the 500 figure as coming from that HTML text, not as a biology rule.
- The "about 1000 points" figure is from Angelopoulos et al., "Uncertainty Sets for Image Classifiers using Conformal Prediction" (https://arxiv.org/abs/2009.14193): they "require only 1000 held-out data points in practice." That is an image-classifier paper. Do not present 1000 as a requirement for this benchmark.
- Conformal risk control (Angelopoulos et al., ICLR 2024; https://openreview.net/forum?id=33XGfHLtZg). Abstract: "the expected value of a monotone loss function is controlled", under exchangeability. This is an expected-loss bound, not a high-probability bound on one finite dataset. Learn-then-Test is the high-probability sibling; it was not re-opened. Do not conflate the two.

**Proposed, used on embedding retrieval including EC, not a CATH split method.** Boger et al., Nature Communications, online 2 Jan 2025, "Functional protein mining with conformal guarantees", DOI 10.1038/s41467-024-55676-y (https://www.nature.com/articles/s41467-024-55676-y). Abstract: conformal risk control on embedding-based retrieval, including EC, "controlling the risk of false identifications." Not Nature Methods. This is the closest published use to a nearest-neighbour protein label transfer. It does not remove the need for a homology split.

CPEC, PLOS Computational Biology, 2024 (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012135). The author list was not recorded, so no authors are named. The abstract was not re-opened in this pass. A previous extract said the abstract mixes "expected FDR" and "with high probability." Do not quote that wording as verified. The useful caution, if the abstract is re-read later, is to keep conformal risk control (expected loss) distinct from a high-probability guarantee.

**Not verified.** Sample-size guidance for histogram binning was not verified (Zadrozny and Elkan 2001 was not opened). Niculescu-Mizil and Caruana, ICML 2005, was not opened. Do not invent a sample-size cutoff for isotonic regression versus Platt scaling. Both are curve-fitting calibrators. Neither gives a finite-sample guarantee by itself.

**Inference, grounded in Angelopoulos et al. 2024, not stated by them for this benchmark.** Conformal risk control is a cleaner way to turn a monotone score (here, nearest-neighbour distance) into a threshold with a finite-sample bound on expected false-discovery or precision loss, but only if calibration and test are exchangeable. Same remoteness is part of that assumption. A calibration set that is less remote than the test set voids the guarantee. Fitting a precision-versus-distance curve on that easier set has the same failure, without the guarantee. Conformal prediction does not fix homology leakage.

## 6. Splits for remote-homology benchmarks (SCOP / CATH)

**Established.**

- TAPE holds out entire evolutionary groups (Rao et al. 2019, abstract only; URL above). That is a class-holdout idea, not an identity-threshold recipe.
- CATH's own redundancy lists use BLAST identity plus an overlap rule (20% or 40%, overlap 60% of the longer sequence; download page above). That is redundancy reduction inside CATH, not a train/test split against an external lookup set.
- CATH assigns homology with structural and HMM evidence below 35% identity (FAQ above). A split that only removes pairs at >=20% identity is weaker than CATH's own notion of a homologous superfamily.

**Not verified.** The CATHsingle "one representative per superfamily" paper (a PMC page was mentioned in an earlier pass) was not re-opened. Authors are not named. An NSF PDF about one query per superfamily was not verified. Do not cite either.

**Inference.** "One domain per superfamily" is a sampling rule that stops one superfamily from dominating the metric. It does not, by itself, separate a query from a lookup set. If the lookup set still contains another member of that superfamily, the query is not remote. Use it as a balance constraint on top of a homology split, not instead of one.

ProteinNet's topology split (profile-profile, about 10%, full pre-thinning set, same logic for validation) is the documented SCOP/CATH-style construction that was actually read. It is the reference to copy if the label is a CATH topology.

## 7. Pitfalls when test queries are deliberately remote from the lookup set

**Established, from the sources above, applied to this setting.**

- Separation against cluster representatives fails by transitivity (ProteinNet methodology page; Teufel et al. 2023 on representative-only criteria). Remote-looking queries can still match a non-representative lookup sequence.
- Domain versus domain misses a domain sitting inside a longer lookup chain (ProteinNet methodology page). For EC or CATH domains against full chains, set coverage so a shared domain counts.
- MMseqs2 default identity is not column percent identity (MMseqs2 wiki). A 20% default-mode cutoff is not the same filter as BLAST 20% over 60% overlap, which is what CATH's own lists use.
- Linclust does not reach 20% (MMseqs2 wiki: down to 50%). Using it "at 20%" does not do what the flag suggests.
- Profile search is more sensitive than pairwise identity, and profile-profile is what ProteinNet uses for topology separation. Pairwise 20% leaves structural neighbours in.
- A prefilter can drop the very pairs a low-identity search was meant to catch (MMseqs2 wiki). Sensitivity parameters have to be high enough that a missed edge is unlikely, and that rate was not quantified here.
- Precision without coverage rewards abstention (CAFA2, Jiang et al. 2016, on coverage under partial evaluation). Remote queries abstain. Report both.
- A calibration set built from easier neighbours is not exchangeable with a remote test set (inference from Angelopoulos et al. 2024 exchangeability, plus ProteinNet's rule that validation uses the same separation as test). Thresholds fit on the easier set will look better than they are.

**Inference.** In a frozen nearest-neighbour setup, leakage and miscalibration compound. A leaked neighbour donates both the label and a small distance, so the query looks confident and correct. Removing that neighbour is the split's job. Calibration only decides when to abstain among queries that are already separated.

## Requirements implied

Each line is a requirement for the CATH v4.4 split. The source is what justifies it, not a claim that the source wrote this sentence.

1. Separate every test and calibration sequence from every sequence in the lookup set before redundancy reduction, not from cluster representatives. Source: ProteinNet methodology page; Teufel et al. 2023 on representative-only criteria.
2. State the identity definition. If the cutoff is meant to match CATH's lists, use identical aligned residues over aligned columns (MMseqs2 `--alignment-mode 3`) or BLAST-style identity, not the MMseqs2 default similarity score. Source: MMseqs2 wiki; CATH download page (BLAST, overlap over the longer sequence).
3. Set coverage explicitly, and count a domain match inside a longer chain. Do not rely on an unread default. Source: MMseqs2 wiki (`--cov-mode` 0/1/2); ProteinNet methodology page (domain-in-chain leakage); CATH download page (60% overlap of the longer sequence).
4. For a topology-level CATH split, do not treat pairwise or even sequence-profile 20% as sufficient. Prefer profile-profile search near 10%, as ProteinNet does, or document that the split is weaker than that. Source: ProteinNet methodology page; CATH FAQ (homology can be assigned at 20% identity plus structure, or by HMM-HMM).
5. Build the calibration set with the same separation, and the same remoteness, as the test set. Do not fit thresholds on an easier neighbour set. Source: ProteinNet methodology page; Angelopoulos et al., ICLR 2024 (exchangeability, for any conformal guarantee).
6. Use the test set once. Fit any distance threshold only on calibration. Source: inference from the three-role design plus the calibration sources above. No benchmark paper opened here states this project's exact three-role protocol.
7. Report coverage together with precision at every threshold. Source: Jiang et al. 2016 (CAFA2 coverage under partial evaluation). The mapping from their coverage definition to an NN abstention rule is inference.
8. If structures exist for both sides, consider a structure-based redundancy pass (Foldseek) in addition to sequence-profile separation, and set coverage explicitly because the documented default disagrees with issue reports. Source: van Kempen et al. 2023/2024; Foldseek issue #299 (2024). This is additional, not a replacement.
9. Do not use CD-HIT below 40% identity, and do not use Linclust for a 20% split. Source: CD-HIT user guide; MMseqs2 wiki.
10. Do not cite a 20% filter as proof that structural homologs are absent. Source: ProteinNet methodology page (twilight zone; topology separation needs profile-profile); CATH FAQ.

## Candidate improvements over plain MMseqs2 filtering

Plain filtering here means: MMseqs2 profile search at a stated 20% identity, against the full pre-reduction lookup set, with identity mode and coverage written down. That already fixes the worst unspecified-default failures. The items below are additions.

| Candidate | What it adds | Effort on one GPU, a few weeks | Benefit | Verdict |
| --- | --- | --- | --- | --- |
| Write down identity mode, coverage, and "against the full set" | Turns an ambiguous 20% into a reproducible filter | Hours | High. Without this, the split cannot be compared with CATH or ProteinNet | Do this. It is specification, not a new tool |
| Graph-Part on a profile-search graph | Removes the representative-only hole and can keep more sequences than clustering-and-discarding | Days, if edges are precomputed. Partitioning itself is CPU graph work, not GPU | Medium, only if the edge list is already the sensitive search. Zero if edges are pairwise 20% and remote homologs were never edges | Adopt as the partitioner, not as the homology detector. Fall back to reduction if it fails to partition (Teufel et al. 2023) |
| Profile-profile search near 10% for topology labels | Matches the only topology-level recipe that was read | Days to a week of CPU search, disk, and memory. Not a GPU training job. May be heavy for a full CATH v4.4 all-vs-all | High if the label is topology or architecture. Low if the claim is only "no 20% profile hit" | Do it for the CATH topology split if the search finishes. If it does not, report the weaker filter instead of implying topology separation |
| Foldseek clustering where structures exist | Catches structural neighbours that sequence identity misses | Moderate if CATH domain structures are already on disk. Useless for sequences with no model | Medium as a second filter. Does not define sequence-profile leakage | Optional second pass. Set `-c` explicitly |
| Conformal risk control for the distance threshold | Finite-sample bound on expected precision loss, if calibration and test are exchangeable | Days to implement and check. Needs a calibration set of the same remoteness. Hundreds of points are the gentle-intro suggestion; 1000 is an image-paper practice note, not a biology rule | High for the threshold, zero for leakage | Use it to set the threshold. Do not use it instead of the split |
| Isotonic or Platt curve on distance | A fitted precision-versus-distance curve | Hours | Lower than conformal risk control: no finite-sample guarantee, same exchangeability failure if the calibration set is easier | Skip unless conformal risk control is too awkward to implement. Sample-size rules for these methods were not verified |
| One-domain-per-superfamily sampling | Stops large superfamilies from dominating the score | Small, once superfamily labels exist | Real for reporting balance. Does not create remoteness | Add as a sampling constraint after the split. The CATHsingle paper was not re-read; do not cite it |

**Recommendation.** Keep MMseqs2 profile search, with an explicit identity definition and coverage mode, against the full pre-reduction lookup set. For a CATH topology claim, move the edge definition toward ProteinNet's profile-profile search near 10% if the compute fits. Use Graph-Part to partition that graph, not to discover remote homologs. Use conformal risk control to choose the abstention threshold on a calibration set that was separated the same way as the test set. Neither tool replaces the profile search. Neither makes a 20% pairwise filter into a topology split.
