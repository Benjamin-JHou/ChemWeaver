# ChemWeaver Git Configuration

## Commit Message Standards

To maintain consistency and avoid privacy issues, all commit messages should follow these guidelines:

### 🎯 **Commit Message Format**
```
[category] Brief descriptive title

Detailed explanation (optional):
- Change 1: What was done
- Change 2: Additional details
- Change 3: Technical specifics

Category tags:
🔧 [feature]     - New functionality
🔨 [fix]        - Bug fixes  
📄 [docs]        - Documentation updates
🚀 [deploy]      - Deployment changes
🧪 [test]        - Testing and validation
🔧 [refactor]   - Code improvement
🏗 [build]       - Build system changes
🎨 [style]       - Code style/formatting
🔒 [security]    - Security fixes
🔖 [perf]        - Performance improvements
```

### 📋 **Category Examples**

✅ Good examples:
```
[feature] Add AI surrogate model integration
[docs] Update installation instructions
[fix] Resolve dependency loading issue
[test] Add independent validation framework
[deploy] Update Docker configuration
[refactor] Improve error handling in pipeline
```

❌ Bad examples (avoid):
```
Fix Benjamin's local setup error
Update skill.md as requested
Remove deployment scripts we don't need
Fix bug in my laptop configuration
```

### 🔒 **Privacy Guidelines**

✅ Do include:
- Technical changes and improvements
- Feature additions and fixes
- Documentation updates
- Testing and validation results
- Performance improvements
- API changes
- Configuration updates

❌ Do NOT include:
- Personal names or references
- Local machine specifics
- Personal development environment details
- Internal testing configurations
- Temporary file paths
- "my laptop", "my machine", etc.

### 📄 **Documentation Updates**

When updating documentation:
```
[docs] Update README.md for publication readiness

📄 Documentation Updates:
✅ Fix DOI reference to match Zenodo publication
✅ Remove outdated references (skill.md, deleted capabilities)  
✅ Simplify documentation links for cleaner presentation
✅ Update citation format to remove unpublished paper claims
```

### 🧪 **Validation Updates**

When adding validation results:
```
[test] Add independent reproducibility validation

🧪 Validation Updates:
✅ Complete independent user simulation
✅ Add Nature-level Figure 6 generation
✅ Include comprehensive validation metrics
✅ Document validation methodology and results
```

### 🏗 **Code Changes**

For core functionality:
```
[feature] Add AI surrogate model integration

🔧 Feature Updates:
✅ Implement multi-modal neural network architecture
✅ Add uncertainty quantification methods
✅ Integrate physics-regularized loss functions
✅ Create decision layer for hit selection
```

### 🎯 **General Guidelines**

1. **Use present tense**: "Add", "Update", "Fix", "Remove"
2. **Be specific**: "Add pipeline" vs "Add stuff"
3. **Focus on what**, not why
4. **Keep first line under 72 characters**
5. **Use conventional commits**: type(scope): subject
6. **No personal references**: Avoid names, locations, machines

### 🚀 **Review Process**

Before committing:
1. Review staged changes with `git status`
2. Check commit message follows standards
3. Ensure no personal information in changes
4. Verify documentation is accurate
5. Test functionality still works

### 🔧 **Git Hooks Configuration (Optional)**

To enforce these standards, consider adding:
```bash
# Setup commit message template
git config commit.template "ChemWeaver"
```

This ensures all commits follow consistent, professional standards suitable for open-source scientific software development.