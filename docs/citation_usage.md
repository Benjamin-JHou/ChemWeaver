# Citing ChemWeaver

## How to Cite ChemWeaver in Your Research

Thank you for using ChemWeaver! Proper citation helps us track the impact of our work and supports continued development.

---

## Citation Formats

### Software Citation (Primary)

When you use ChemWeaver software in your research, please cite:

**BibTeX**:
```bibtex
@software{chemweaver_2024,
  title = {ChemWeaver: Reproducible AI-Driven Virtual Screening Infrastructure},
  author = {Hou, Benjamin J. and {ChemWeaver Development Team}},
  year = {2024},
  url = {https://github.com/Benjamin-JHou/ChemWeaver},
  version = {1.0.3},
  doi = {10.5281/zenodo.18497305}
}
```

**APA**:
```
Hou, B. J., & ChemWeaver Development Team. (2024). ChemWeaver: 
Reproducible AI-Driven Virtual Screening Infrastructure (Version 1.0.3) 
[Computer software]. https://github.com/Benjamin-JHou/ChemWeaver
```

**MLA**:
```
Hou, Benjamin J., et al. "ChemWeaver: Reproducible AI-Driven Virtual 
Screening Infrastructure." GitHub, 2024, 
https://github.com/Benjamin-JHou/ChemWeaver.
```

### Associated Publication (When Available)

If you want to cite the scientific methodology paper:

**BibTeX**:
```bibtex
@article{chemweaver_2024_nature,
  title={ChemWeaver: A Reproducible AI-Augmented Infrastructure for 
         Prospective Virtual Screening and Translational Drug Discovery},
  author={Hou, Benjamin J. and [Co-authors]},
  journal={Nature Methods},
  year={2024},
  volume={XX},
  issue={XX},
  pages={XXX--XXX},
  publisher={Nature Publishing Group}
}
```

**Note**: The Nature Methods paper is currently in preparation. Until publication, please cite the software above.

---

## What to Cite

### You MUST Cite ChemWeaver If:

1. **You use ChemWeaver for virtual screening** in your research
2. **You use ChemWeaver predictions** as part of your analysis
3. **You build upon ChemWeaver code** for your own tools
4. **You use ChemWeaver benchmarks** for method comparison
5. **You use ChemWeaver uncertainty quantification** methods

### You SHOULD Mention ChemWeaver If:

1. You use ChemWeaver for initial exploration or pilot studies
2. You adapt ChemWeaver code examples for your work
3. You use ChemWeaver documentation for educational purposes

---

## Version-Specific Citation

If you need to cite a specific version:

```bibtex
@software{chemweaver_v1_0_0,
  title = {ChemWeaver: Reproducible AI-Driven Virtual Screening Infrastructure},
  author = {Hou, Benjamin J. and {ChemWeaver Development Team}},
  year = {2024},
  month = {2},
  day = {5},
  url = {https://github.com/Benjamin-JHou/ChemWeaver},
  version = {1.0.3},
  doi = {10.5281/zenodo.18497305}
}
```

**Note**: This DOI corresponds to the current software release.

---

## Citing Specific Features

### AI Surrogate Model

If you specifically use the AI surrogate prediction:

```bibtex
@software{chemweaver_ai_2024,
  title = {ChemWeaver AI Surrogate: Uncertainty-Aware Docking Score Prediction},
  author = {Hou, Benjamin J.},
  year = {2024},
  url = {https://github.com/Benjamin-JHou/ChemWeaver},
  note = {Part of ChemWeaver v1.0.3}
}
```

### Uncertainty Quantification

If you use ChemWeaver's uncertainty quantification methods:

```bibtex
@software{chemweaver_uq_2024,
  title = {ChemWeaver Uncertainty Quantification: Deep Ensembles and 
           Evidential Learning for Virtual Screening},
  author = {Hou, Benjamin J.},
  year = {2024},
  url = {https://github.com/Benjamin-JHou/ChemWeaver},
  note = {Part of ChemWeaver v1.0.3}
}
```

### Reproducibility Framework

If you use ChemWeaver's reproducibility features:

```bibtex
@software{chemweaver_repro_2024,
  title = {ChemWeaver Reproducibility Framework: Container-Native Virtual 
           Screening with Cryptographic Verification},
  author = {Hou, Benjamin J.},
  year = {2024},
  url = {https://github.com/Benjamin-JHou/ChemWeaver},
  note = {Part of ChemWeaver v1.0.3}
}
```

---

## Citation in Publications

### Methods Section Example

```latex
\section{Methods}

\subsection{Virtual Screening}

Virtual screening was performed using ChemWeaver v1.0.3 
\cite{chemweaver_2024}, a reproducible AI-driven screening 
infrastructure. The screening pipeline consisted of four stages: 
(1) compound standardization, (2) property filtering using Lipinski's 
Rule of Five, (3) AI surrogate prediction with uncertainty 
quantification, and (4) hit selection based on confidence thresholds.

The AI surrogate model predicted binding affinity with associated 
uncertainty estimates using a deep ensemble of five neural networks. 
Compounds with confidence scores $\geq 0.7$ and uncertainty $\leq 0.5$ 
were selected as hits for experimental validation.
```

### Results Section Example

```latex
\section{Results}

\subsection{Hit Identification}

Using ChemWeaver \cite{chemweaver_2024}, we screened 1,000,000 compounds 
against Target X. The AI surrogate model processed the library in 
45 minutes, identifying 47 high-confidence hits (confidence $\geq 0.7$). 
The model showed good calibration with an Expected Calibration Error 
(ECE) of 0.042.

Experimental validation of the top 20 hits resulted in 8 confirmed 
actives (40\% hit rate), demonstrating the utility of ChemWeaver's 
uncertainty-aware predictions.
```

---

## Including ChemWeaver in Acknowledgments

If ChemWeaver was helpful but not central to your research:

```
We thank the ChemWeaver development team for providing the 
open-source virtual screening infrastructure used in preliminary 
analysis (Hou et al., 2024).
```

---

## Tracking ChemWeaver Citations

### Google Scholar

Add ChemWeaver to your Google Scholar profile to track citations:

1. Go to [Google Scholar](https://scholar.google.com)
2. Click "My Citations"
3. Add publication with:
   - Title: "ChemWeaver: Reproducible AI-Driven Virtual Screening Infrastructure"
   - Authors: "Hou, Benjamin J."
   - Year: 2024
   - URL: https://github.com/Benjamin-JHou/ChemWeaver

### ResearchGate

Share your ChemWeaver-based research on ResearchGate:

1. Upload your paper
2. Tag with "ChemWeaver"
3. Link to ChemWeaver repository

---

## Dependencies to Also Cite

ChemWeaver builds upon many excellent open-source projects. Please also cite:

### Essential Dependencies

**RDKit** (Cheminformatics):
```bibtex
@misc{rdkit,
  author = {RDKit},
  title = {RDKit: Open-source cheminformatics},
  url = {https://www.rdkit.org}
}
```

**NumPy** (Numerical computing):
```bibtex
@article{numpy,
  author = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, 
            Stéfan J. et al.},
  title = {Array programming with NumPy},
  journal = {Nature},
  year = {2020},
  volume = {585},
  pages = {357--362}
}
```

**scikit-learn** (Machine learning):
```bibtex
@article{sklearn,
  author = {Pedregosa, Fabian and Varoquaux, Gaël and Gramfort, Alexandre 
            et al.},
  title = {Scikit-learn: Machine Learning in Python},
  journal = {Journal of Machine Learning Research},
  year = {2011},
  volume = {12},
  pages = {2825--2830}
}
```

### Optional Dependencies

**PyTorch** (If using neural network components):
```bibtex
@inproceedings{pytorch,
  author = {Paszke, Adam and Gross, Sam and Massa, Francisco et al.},
  title = {PyTorch: An Imperative Style, High-Performance Deep 
           Learning Library},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019}
}
```

---

## Citation FAQ

### Q: Do I need to cite ChemWeaver if I only use it for a figure?

**A**: Yes, if ChemWeaver contributed meaningfully to your research, please cite it. If it's a minor component, acknowledgment is sufficient.

### Q: How do I cite ChemWeaver if I modified the code?

**A**: Cite the original software and mention your modifications in your methods section:
```
Virtual screening was performed using a modified version of 
ChemWeaver v1.0.3 (Hou et al., 2024), with custom feature 
extraction added to the standard pipeline.
```

### Q: What if I use ChemWeaver with other tools?

**A**: Cite all tools that significantly contributed to your research. For example:
```
We used an integrated workflow combining ChemWeaver \cite{chemweaver_2024} 
for AI-based screening and AutoDock Vina \cite{trott2010autodock} for 
structure-based validation.
```

### Q: Should I cite the software or the paper?

**A**: 
- **Now**: Cite the software (v1.0.3)
- **After publication**: Cite the Nature Methods paper as primary, software as secondary

### Q: How do I get the Zenodo DOI?

**A**: The current software DOI is already assigned: `10.5281/zenodo.18497305`.

---

## Reporting Your Publication

Help us track ChemWeaver's impact by letting us know about your publication:

**How to report**:
1. Open a GitHub Issue with tag "publication"
2. Include citation details in the issue body
3. Link the publication DOI and venue

**Information to include**:
- Publication title
- Authors
- Journal/Conference
- DOI (if available)
- How ChemWeaver was used
- Key findings

---

## Citation Metrics

We track:
- GitHub stars and forks
- Zenodo downloads
- Google Scholar citations
- Publication mentions

**Current Metrics** (as of v1.0.3):
- GitHub Stars: Track live values on the GitHub repository page
- Zenodo Downloads: Track live values on the Zenodo record page
- Citations: Track live values in Google Scholar

---

## Version History

| Version | Date | DOI | Changes |
|---------|------|-----|---------|
| 1.0.3 | 2026-02-16 | 10.5281/zenodo.18497305 | Metadata and reproducibility updates |

---

## Questions?

For citation questions or to report your publication:

- **GitHub**: [Open an issue](https://github.com/Benjamin-JHou/ChemWeaver/issues)
- **Repository**: https://github.com/Benjamin-JHou/ChemWeaver
- **Zenodo**: https://doi.org/10.5281/zenodo.18497305

---

**Thank you for citing ChemWeaver!**

Your citations help us:
- Demonstrate impact for funding
- Improve the software
- Build a community
- Advance open science

---

**Last Updated**: 2026-02-16  
**Citation Version**: 1.0.3  
**Valid Until**: Next major release
