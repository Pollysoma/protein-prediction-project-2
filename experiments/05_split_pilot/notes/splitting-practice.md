# Splitting practice for a CATH nearest-neighbour benchmark

Scope: how this field assigns proteins to splits, not the two named recipes. Each claim is tagged **common practice with a citation**, **recommended by a named source**, or **your own inference**. A quotation is used only where the wording was retrieved. Anything not re-opened is marked unfinished and is not treated as established.

Project figures used below, not a paper: CATH v4.4 has 123,251 sequence-unique domains, 6,630 superfamilies, 4,339 of them with at least two members.

## 1. How are proteins assigned to splits?

- **Common practice with a citation.** A random split of redundant domains is not accepted. Petti and Eddy, PLOS Computational Biology 2022, https://doi.org/10.1371/journal.pcbi.1009492: "A random split is insufficient". The correction (https://doi.org/10.1371/journal.pcbi.1010971) is not the paper.
- **Common practice with a citation.** GraphPart states the naive default and then rejects it. Teufel et al., NAR Genomics and Bioinformatics 2023, https://doi.org/10.1093/nargab/lqad088, https://pmc.ncbi.nlm.nih.gov/articles/PMC10578201/: "Typically, this can be achieved by splitting the data randomly." Next sentence, shortened: "for biological sequences this approach is severely flawed".
- **Recommended by a named source.** Same paper: do not cluster and then split. Clustering tools "should not be used for homology partitioning", because the "maximum similarity criterion is only applied to the representative sequences". Also: "CD-HIT and MMseqs2 are designed for fast clustering and not for homology reduction". Their concrete failure mode: "A threshold of 30% identity is applied within the clusters but not between them".
- **Common practice with a citation.** The order GraphPart argues for is reduction first, assignment second. Remove pairs above the cutoff, then split what remains. That is their method, not a count of how many benchmarks do it.
- **Recommended by a named source.** DOME separates the two steps rather than treating random assignment as enough. Walsh et al., Nature Methods 2021, https://doi.org/10.1038/s41592-021-01205-4: "initial data should be divided randomly" and, separately, "reducing the number of homologs".
- **Common practice with a citation.** CATHe does both. Nallapareddy et al., Bioinformatics 2023, https://doi.org/10.1093/bioinformatics/btad029, https://pmc.ncbi.nlm.nih.gov/articles/PMC9887088/: "we randomly split the data into training, validation, and testing sets". Supplementary S2 then uses MMseqs2 to cut "sequence identity between training and the other two (testing and validation) sets" "to less than 20%", and BLAST to "remove homologous sequences, at 20% sequence identity, between the three sets".
- **Common practice with a citation.** Remote claims use group holdout, not a random domain draw. ProtTucker, Heinzinger et al., NAR Genomics and Bioinformatics 2022, https://doi.org/10.1093/nargab/lqac043, https://pmc.ncbi.nlm.nih.gov/articles/PMC9188115/: "every homologous superfamily appeared maximally once in test300 ∩ val200".
- **Your own inference.** A random split is acceptable only after the homology constraint has already been enforced. It is not acceptable on CATH v4.4 as released. Family or class balance does not replace that constraint.

## 2. What is the assignment stratified by, when anyone stratifies?

- **Not found, in the sources opened here, as a CATH-split standard:** resolution, sequence length, taxonomy, or annotation quality. Sensible. Not something these benchmarks were shown to stratify on.
- **Unfinished.** An earlier pass said GraphPart tries to keep class balance, and that DOME asks whether class distributions differ. Those lines were not re-opened. They are not cited as verified.
- **Common practice with a citation, and only as a sampling rule.** ProtTucker's one-domain-per-superfamily constraint (quote in section 1) is not a general stratification standard. It fixes which groups may contribute a query.
- **Your own inference.** One domain per superfamily makes the query set follow superfamily counts per class, not domain counts. A class with many small superfamilies is then over-weighted relative to a class whose domains sit in a few large superfamilies. Report both distributions. Do not call the first one "class-balanced" without saying which denominator was used.

## 3. What is the single representative?

- **Common practice with a citation.** GraphPart, describing CD-HIT and pointing at Li and Godzik, Bioinformatics 2006, https://doi.org/10.1093/bioinformatics/btl158: "the longest sequence becomes the representative of the first cluster." The 2006 PDF was not re-opened, so that sentence is GraphPart's wording, not a confirmed quotation from Li and Godzik.
- **Unfinished.** Linclust (Steinegger and Söding, Nature Communications 2018, https://doi.org/10.1038/s41467-018-04964-5) and the MMseqs2 user-guide defaults for `--cluster-mode` were not re-opened. No default is stated here. Do not fill the gap from memory or from a GitHub issue that was not opened.
- **Unfinished.** ProteinNet's representative rule was not re-opened. An earlier pass said the GitHub note uses cluster centres and does not select by structure quality. That is not repeated as verified. https://github.com/aqlaboratory/proteinnet/blob/master/docs/splitting_methodology.md
- **Your own inference.** Longest-sequence representatives favour long chains and, on full proteins, multi-domain chains. For a frozen sequence embedding, crystallographic resolution does not change the vector, so "best resolution" is a structure-quality choice with no effect on the embedding. If the draw is random, publish the seed. If it is longest, say so and show the length shift. The database's own S-level representative is not, on the evidence opened here, a substitute for an identity filter against the other roles. No S-level guarantee was re-opened, so none is claimed.

## 4. How large should calibration be, relative to test?

- **Not found.** No protein-benchmark standard for the calibration:test ratio. A preprint claim that training sets of 60–80% are common was not re-opened and is not cited.
- **Common practice with a citation.** One published table, not a rule. CATHe Table 1 (Nallapareddy et al. 2023, PMC above): training 30,837, validation 6,863, testing 6,862. The other columns of that table were not interpreted. Section 2.3.2 says the architecture was optimized on the testing set. That is a warning, not a model to copy.
- **Common practice with a citation.** ProtTucker uses 200 validation domains and 300 test domains, then 219 test domains after dropping queries with HVAL>0 against the lookup (Heinzinger et al. 2022). A design choice. Not a ratio.
- **Recommended by a named source, sentence not re-opened.** DOME's point, as checked in an earlier pass, is that evaluation data are not for tuning. The phrases retrieved this pass are only the random-division and homolog-reduction lines in section 1. Treat "fit the threshold off the test set" as the recommendation we are following, and do not pretend a Box 1 line was re-quoted.
- **Your own inference, from a named bound.** Hanley and Lippman-Hand, JAMA 1983, https://jamanetwork.com/journals/jama/article-abstract/386759: if a bad event is seen zero times in n trials, the approximate 95% upper bound is 3/n. The PDF was not re-opened, so this is the stated rule, not a verbatim line. The binding n is the number of accepted transfers at the operating point, not the size of the calibration set. Arithmetic, labelled inference: a claim of at most 5% errors, with none observed, needs about 60 accepted calls (3/0.05). Below that, the interval cannot support the claim however large the unused part of the set is.
- **Recommended by a named source, paper not re-opened.** Brown, Cai and DasGupta, Statistical Science 2001, https://doi.org/10.1214/ss/1009213286, are the usual warning that Wald intervals fail near 0 and 1. Use Wilson or Clopper–Pearson. No sentence from that paper is quoted.
- **Your own inference.** Clinical sample-size papers (Riley and others) do not transfer to this benchmark. They are not cited.

## 5. Must calibration and test be mutually non-redundant?

- **Not a named rule.** No guideline opened here says the calibration split and the test split must be homology-reduced to each other. Whether most benchmarks ignore that pair is not something this pass counted. The absence of a named rule is not a reason to skip the check.
- **Common practice with a citation.** CATHe does not ignore it. BLAST removes homologs "at 20% sequence identity, between the three sets" (supplementary S2, Nallapareddy et al. 2023). That is between sets, not a second reduction inside each set.
- **Common practice with a citation.** ProtTucker constrains superfamily membership across val200 and test300, then filters test against the lookup by HVAL. The sentences retrieved do not describe a pairwise identity filter between validation and test.
- **Your own inference.** Read literally, "test300 ∩ val200" is an intersection of two disjoint domain sets, so the constraint would be empty. The operational reading is at most one domain per superfamily across the combined query sets. Different superfamilies can still share local sequence similarity. Superfamily exclusion is not sequence exclusion.
- **Recommended by a named source.** DOME, in the phrases retrieved, asks whether train and test are independent. It does not require a homology filter between validation and test.

## 6. What has to be reported?

- **Recommended by a named source.** Nature Communications computational reporting guidelines, https://www.nature.com/documents/Computational_tools_reporting_guidelines.pdf: "the train-test-validation splits have to be reported in detail." The phrase was confirmed by search. The PDF was not re-opened.
- **Recommended by a named source.** Heil et al., Nature Methods 2021, https://doi.org/10.1038/s41592-021-01256-7. Silver requires that "all random components in the analysis are set to be deterministic." A seed that is not recorded fails that bar.
- **Recommended by a named source, partly unfinished.** DOME (Walsh et al. 2021) is the checklist people cite for split reporting. This pass retrieved "initial data should be divided randomly" and "reducing the number of homologs". An earlier pass also reported Box 1 lines ("How were the sets split?"; redundancy "to less than X% pairwise identity"; "Are the data, including the data splits used, released"). Those three lines were not re-opened. Use DOME as the standard, and do not put the unreopened lines in quotation marks.
- **Common practice with a citation.** CATHe releases the sets. Zenodo, https://zenodo.org/record/6327572.
- **Your own inference.** A manifest checksum is worth publishing. It is engineering, not a rule from CAFA, CASP, or a journal checklist opened here. CASP withholds structures; it is not an ML split checklist, and no CASP DOI is invented. CAFA's split is temporal. The sentence about computing maximum F-measure on the evaluation set was not re-opened and is not cited.

## 7. How do remote-homology benchmarks make the test remote?

Three constructions showed up. **Your own inference:** that is not one standard. It is what these papers did.

- **Common practice with a citation. Filter the lookup against the queries.** ProtTucker: "the data set had been optimized for minimal performance of MMseqs2." Queries that still hit the lookup at HVAL>0 are dropped, 300 to 219 (Heinzinger et al. 2022). Exact flags (`--num-iterations 3`, `-s 7.5`, bidirectional coverage) were not re-opened.
- **Common practice with a citation. Remove library sequences similar to both held-out sets.** CATHe, quotes in section 1: MMseqs2 at 20% between training and the other two sets, then BLAST at 20% between all three.
- **Common practice with a citation. Hold out whole groups.** ProtTucker's one-superfamily rule, quoted in section 1. DeepSF's three-level holdout was not re-read (Hou et al., Bioinformatics 2018, https://doi.org/10.1093/bioinformatics/btx780). A caption is not a protocol. Omitted.
- **Not cited.** Söding 2005 and HH-suite `pdbfilter.pl` were not re-opened.
- **Your own inference.** Filtering the lookup answers whether a neighbour remains after close sequences are gone from the library. Holding out the superfamily answers whether the label can transfer when that superfamily was never seen. Doing both, as ProtTucker does, is a different and stricter claim. It also makes a later MMseqs2 baseline look weak. ProtTucker already said the set was optimized against MMseqs2. A reviewer who misses that sentence will call the baseline unfair.

## What I would propose for a CATH v4.4 three-way split

Use the sequence-unique CATH v4.4 domains. Do not treat an S-level list as the remoteness filter.
Draw calibration and test as one domain per superfamily, on disjoint superfamilies, with a fixed seed. Name the representative rule in the manifest. Do not use resolution.
Lookup is the remainder, including other members of a query superfamily unless that query is dropped.
Remove lookup hits to either query set at 20% with the same MMseqs2 iterative profile search the recipe uses. Drop any query that still hits the lookup or the other query set.
Fit the distance threshold only on calibration, at a pre-declared precision. Use test once. Report accepted calls at that point, not just set size. Under the rule of three, about 60 error-free accepted calls are needed before a 5% error claim.
Publish domain ID, role, superfamily, class, length, the exact command, the seed, and a checksum. Plot class counts and length for all three roles, with both domain and superfamily denominators.
Say the estimand in the first paragraph: transfer to a held-out superfamily after close lookup sequences are removed. It is not performance on a random domain.

## Things our supervisor is likely to object to

- The lookup filter is built with MMseqs2, then a reviewer will want an MMseqs2 baseline. ProtTucker already admitted the set was optimized against that tool. Profile search at 20% can also remove remote homologs the method should still be allowed to find. Those are different objections. Answer both, or the filter looks like it was chosen to flatter the embedding.
- One domain per superfamily throws away most of the internal diversity of the 4,339 multi-member superfamilies and changes the estimand. CATHe, after its homology cuts, still drew a random domain split (section 1). GraphPart's objection still stands if that random split is done by clustering first.
- A homology-reduced calibration set, with the threshold frozen before test, is stricter than scoring the operating point on the evaluation set. It will look unusual next to CAFA-style reporting. It is the right choice for a threshold we claim was not fit on test. Say that before being asked.
- Disjoint superfamilies do not imply disjoint sequences. Two superfamilies can share a local alignment. CATHe's between-set BLAST step is the part a supervisor can demand and that ProtTucker's wording does not clearly include.
- The representative rule is undefended until we measure it. Longest shifts length. A set-cover centre, if that is what the clustering tool emits, shifts toward sequences with many alignments. MMseqs2's default was not re-opened, so it must not be asserted in the memo.
- There is no temporal holdout in this proposal. CAFA and ProteinNet are often described as holding out later targets; those protocols were not re-opened here, so they are not cited as verified. CATH labels are classifications of structures that already exist, so a time cut would not by itself remove homology. That last sentence is inference. The fair reply is that the claim is remote transfer, not prospective discovery. Do not pretend a time cut was tested.
- Several sentences a supervisor may ask to see were not re-opened: MMseqs2 cluster-mode defaults, Linclust's representative sentence, DOME Box 1 lines, ProteinNet's validation counts, DeepSF's three-level protocol, and the Wald-interval paper. They stay out of the memo until someone opens the page.
