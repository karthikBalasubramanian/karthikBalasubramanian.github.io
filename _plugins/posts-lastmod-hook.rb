#!/usr/bin/env ruby
#
# Check for changed posts (from chirpy-starter). Skipped when there is no git
# history, e.g. inside the local Docker build where .git is not copied.

Jekyll::Hooks.register :posts, :post_init do |post|
  next unless system('git rev-parse --is-inside-work-tree > /dev/null 2>&1')

  commit_num = `git rev-list --count HEAD "#{ post.path }"`

  if commit_num.to_i > 1
    lastmod_date = `git log -1 --pretty="%ad" --date=iso "#{ post.path }"`
    post.data['last_modified_at'] = lastmod_date
  end
end
