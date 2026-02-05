#!/bin/bash
# GitHub Push Script with Token
# Run from chemweaver-release directory

cd "$(dirname "$0")"

echo "🔐 Configuring GitHub authentication..."

# Set remote URL with token
git remote set-url origin "https://Benjamin-JHou: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@github.com/Benjamin-JHou/ChemWeaver.git"

echo "✅ Authentication configured"
echo ""
echo "📤 Pushing to GitHub..."
echo ""

# Push main branch
echo "➡️  Pushing main branch..."
git push origin main

if [ $? -eq 0 ]; then
    echo "✅ Main branch pushed successfully!"
else
    echo "❌ Failed to push main branch"
    exit 1
fi

echo ""
echo "🏷️  Pushing v1.0.0 tag..."
git push origin v1.0.0

if [ $? -eq 0 ]; then
    echo "✅ Tag v1.0.0 pushed successfully!"
else
    echo "❌ Failed to push tag"
    exit 1
fi

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo ""
echo "Repository: https://github.com/Benjamin-JHou/ChemWeaver"
echo "Release: https://github.com/Benjamin-JHou/ChemWeaver/releases/tag/v1.0.0"
echo ""
echo "Next steps:"
echo "1. Create GitHub Release (triggers Zenodo DOI)"
echo "2. Update README with Zenodo badge"
echo "3. Announce on social media"
