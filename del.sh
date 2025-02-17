# Save the current branch name
current_branch=$(git symbolic-ref --short HEAD)

# List all branches and loop through each one to remove *.ply files
for branch in $(git branch --format '%(refname:short)'); do
    git checkout "$branch"        # Checkout the branch
    git rm --cached *.ply         # Remove .ply files from the index
    git commit -m "Remove *.ply files from $branch"
done

# Checkout back to the original branch
git checkout "$current_branch"
