# frozen_string_literal: true
#
# Make TeX math in Markdown posts survive kramdown so Chirpy's MathJax can render it.
#
# The old Jalpc theme loaded MathJax 2 on every page with `$...$` inline and
# `$$...$$` display delimiters, and posts were written that way. Chirpy only
# loads MathJax when a page sets `math: true`, and kramdown does not know about
# single-dollar math, so it treats the TeX inside `$...$` as Markdown:
#   - `\{ \}  \|  \\` lose their backslash (sets lose braces, norms become |x|),
#   - `|` splits the line into a table, `_` / `*` become emphasis,
#   - quotes and `--` / `...` get "smart" typography.
# And a `$$...$$` line that is not separated from the surrounding text by blank
# lines is parsed as *inline* math, so display equations run into the paragraph.
#
# This generator runs before rendering and, for every post (outside fenced or
# indented code and inline code spans):
#   1. backslash-escapes Markdown-special characters inside `$...$`, so that
#      kramdown's own unescaping hands MathJax the original TeX unchanged;
#   2. puts blank lines around stand-alone `$$` display blocks so kramdown emits
#      them as display math (`\[...\]`);
#   3. sets `math: true` when the post contains math (unless it sets `math: false`),
#      so new posts don't need to remember the flag.

module MathProtect
  INLINE_MATH = /(?<![\\$\w])\$(?![\s$])((?:\\.|[^$\\\n])+?)(?<!\s)\$(?![$\w])/.freeze
  FENCE = /\A\s*(`{3,}|~{3,})/.freeze
  LIST_ITEM = /\A\s*(?:[*+-]|\d+[.)])\s+/.freeze

  module_function

  def escape_set(gfm)
    gfm ? /[\\.*_+`<>()\[\]{}#!:|"'$=~-]/ : /[\\.*_+`<>()\[\]{}#!:|"'$=-]/
  end

  # Escape Markdown specials inside inline math, skipping inline code spans.
  def protect_inline(line, specials)
    found = false
    parts = line.split(/(`+[^`]*?`+)/)
    out = parts.map do |part|
      next part if part.start_with?('`')

      part.gsub(INLINE_MATH) do
        found = true
        "$#{Regexp.last_match(1).gsub(specials) { |c| "\\#{c}" }}$"
      end
    end
    [out.join, found]
  end

  def blank?(line)
    line.strip.empty?
  end

  def process(content, gfm)
    specials = escape_set(gfm)
    lines = content.split("\n", -1)
    out = []
    fence = nil
    in_display = false
    in_list = false
    in_indented_code = false
    has_math = false

    lines.each_with_index do |line, i|
      # Fenced code: copy verbatim.
      if fence
        out << line
        fence = nil if line.strip.start_with?(fence) && line.strip.delete(fence[0]).empty?
        next
      end
      if (m = FENCE.match(line))
        fence = m[1]
        out << line
        next
      end

      stripped = line.strip

      # Inside a multi-line $$ ... $$ block: copy verbatim (kramdown keeps it raw).
      if in_display
        out << line
        if stripped.end_with?('$$')
          in_display = false
          out << '' unless i + 1 >= lines.size || blank?(lines[i + 1])
        end
        next
      end

      # Indented code blocks (4+ spaces after a blank line, outside lists).
      indent = line[/\A[ \t]*/].gsub("\t", '    ').size
      if blank?(line)
        out << line
        next
      end
      in_indented_code = false if indent < 4
      if indent >= 4 && !in_list && (in_indented_code || (i.positive? && blank?(lines[i - 1])))
        in_indented_code = true
        out << line
        next
      end
      if LIST_ITEM.match?(line)
        in_list = true
      elsif indent.zero?
        in_list = false
      end

      # Stand-alone display math line(s).
      if stripped.start_with?('$$')
        has_math = true
        prev_blank = out.empty? || blank?(out.last)
        out << '' unless prev_blank
        out << line
        if stripped.length > 4 && stripped.end_with?('$$') && stripped != '$$'
          out << '' unless i + 1 >= lines.size || blank?(lines[i + 1])
        else
          in_display = true
        end
        next
      end

      new_line, found = protect_inline(line, specials)
      has_math ||= found || line.include?('$$')
      out << new_line
    end

    [out.join("\n"), has_math]
  end
end

Jekyll::Hooks.register :posts, :pre_render do |post, _payload|
  next unless post.extname =~ /\.(md|markdown)\z/i

  input = post.site.config.dig('kramdown', 'input').to_s
  content, has_math = MathProtect.process(post.content, input.casecmp('GFM').zero?)
  post.content = content
  post.data['math'] = true if has_math && post.data['math'].nil?
end
