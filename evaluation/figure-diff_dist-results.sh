# tmp version!

# Combines 3 results figures

#ls -1 *diff_dist_all.pdf
#twitter_bert_768_diff_dist_all.pdf
#twitter_bow_50_diff_dist_all.pdf
#twitter_bow_768_diff_dist_all.pdf


# http://manpages.ubuntu.com/manpages/precise/man1/pdfcrop.1.html
# --margins "<left> <top> <right> <bottom>"
pdfcrop --margins '0 0 4 10' twitter_bert_768_diff_dist_all.pdf twitter_bert_768_diff_dist_all.tmp.pdf
pdfcrop --margins '0 0 0 10' twitter_bow_50_diff_dist_all.pdf   twitter_bow_50_diff_dist_all.tmp.pdf
pdfcrop --margins '0 0 4 10' twitter_bow_768_diff_dist_all.pdf  twitter_bow_768_diff_dist_all.tmp.pdf

# https://github.com/rrthomas/pdfjam#documentation
pdfjam \
twitter_bert_768_diff_dist_all.tmp.pdf \
twitter_bow_50_diff_dist_all.tmp.pdf \
twitter_bow_768_diff_dist_all.tmp.pdf \
--nup 2x1 --landscape --outfile diff_dist_all.tmp.pdf

pdfcrop --margins '0 0 0 0' diff_dist_all.tmp.pdf diff_dist_all.pdf

rm *.tmp.pdf
