#!/bin/bash
# GitHub Push Script
# Run from chemweaver-release directory
# IMPORTANT: Set GITHUB_TOKEN environment variable before running

cd "$(dirname "$0")"

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ Error: GITHUB_TOKEN environment variable not set"
    echo "Please set it with: export GITHUB_TOKEN=your_token_here"
    exit 1
fi

echo "🔐 Configuring GitHub authentication..."

# Set remote URL with token from environment
git remote set-url origin "https://Benjamin-JHou:${GITHUB_TOKEN}@github.com/Benjamin-JHou/ChemWeaver.git"

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
echo "🎉 PUSH COMPLETE!"
echo ""
echo "Repository: https://github.com/Benjamin-JHou/ChemWeaver"
