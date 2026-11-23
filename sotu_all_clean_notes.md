# sotu_all_clean.csv

The State of the Union collection, 1790 to 2026: 240 speeches, one per row. It is a cleaned and extended
version of `sotu_all_new.csv`, which stays as it is because the course notebooks use it.

Columns: `rownames`, `speechtext`, `year`, `date` (as in `sotu_all_new.csv`), plus
`president`, `delivery` (spoken or written) and `type`.

## What changed

- **Five dates corrected** (checked against the American Presidency Project and Gutenberg):
  1817-12-12 is 1817-12-02 (Monroe); 1893-12-03, 1894-12-02, 1895-12-07 and 1896-12-04 are
  1893-12-04, 1894-12-03, 1895-12-02 and 1896-12-07 (Cleveland).
- **Text that the president did not say removed:** signatures at the end of 42 written messages
  (for example "GO. WASHINGTON"), editorial notes after seven Reagan speeches ("NOTE: The President spoke at ..."),
  four editorial footnotes, audience reactions such as [Laughter] and [Applause], and page markers such as [p.2].
- **Typography made consistent:** straight apostrophes and quotation marks everywhere, fractions such as
  "3 1/2" instead of "3\1/2\", no tabs or double spaces.
- **Four speeches added** from the American Presidency Project (https://www.presidency.ucsb.edu):
  Biden 2023-02-07 and 2024-03-07, Trump 2025-03-04 and 2026-02-24. Only the President's words are kept;
  interruptions by members of Congress and the audience, and the transcribers' notes, are left out.

## Things to keep in mind

- **Not every speech is a State of the Union.** In their first year, presidents address Congress instead
  (1989, 1993, 2001, 2009, 2017, 2021, 2025), and 2001-09-20 is Bush's address after the attacks of 11 September.
  The column `type` marks them.
- **Written and spoken messages are mixed.** From 1801 to 1912, and in some later years, the message was a written
  report, often much longer than a speech (Carter's written message of 1981 has over 33,000 words). Compare
  word counts and frequencies over time with this in mind; the column `delivery` marks them.
- **Some years have two speeches** (1790, 1953, 1961, 2001), and 1933 has none. Group by `date`, not `year`,
  to keep each speech apart.
- **Some written messages contain statistical tables** turned into running text (for example 1864, 1883, 1896,
  1910, 1932), and section headings in capitals. These were left as they are.
- Editors' insertions in square brackets that explain a word, such as [the States], and the three
  [missing text] markers in 1911, were left as they are.
