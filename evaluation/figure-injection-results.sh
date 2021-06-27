# Combines 6 injection results figures

#ls -1 *induction_all.pdf
#amazon_bert_768_drift_induction_all.pdf
#amazon_bow_50_drift_induction_all.pdf
#amazon_bow_768_drift_induction_all.pdf
#twitter_bert_768_drift_induction_all.pdf
#twitter_bow_50_drift_induction_all.pdf
#twitter_bow_768_drift_induction_all.pdf

# http://manpages.ubuntu.com/manpages/precise/man1/pdfcrop.1.html
# --margins "<left> <top> <right> <bottom>"
pdfcrop --margins '0 0 4 10' amazon_bow_50_drift_induction_all.pdf    amazon_bow_50_drift_induction_all.tmp.pdf
pdfcrop --margins '0 0 0 10' twitter_bow_50_drift_induction_all.pdf   twitter_bow_50_drift_induction_all.tmp.pdf
pdfcrop --margins '0 0 4 10' amazon_bow_768_drift_induction_all.pdf   amazon_bow_768_drift_induction_all.tmp.pdf
pdfcrop --margins '0 0 0 10' twitter_bow_768_drift_induction_all.pdf  twitter_bow_768_drift_induction_all.tmp.pdf
pdfcrop --margins '0 0 4 0' amazon_bert_768_drift_induction_all.pdf  amazon_bert_768_drift_induction_all.tmp.pdf
pdfcrop --margins '0 0 0 0' twitter_bert_768_drift_induction_all.pdf twitter_bert_768_drift_induction_all.tmp.pdf

# https://github.com/rrthomas/pdfjam#documentation
pdfjam \
amazon_bow_50_drift_induction_all.tmp.pdf \
twitter_bow_50_drift_induction_all.tmp.pdf \
amazon_bow_768_drift_induction_all.tmp.pdf \
twitter_bow_768_drift_induction_all.tmp.pdf \
amazon_bert_768_drift_induction_all.tmp.pdf \
twitter_bert_768_drift_induction_all.tmp.pdf \
--nup 2x3 --landscape --outfile drift_induction_all.tmp.pdf

pdfcrop --margins '0 0 0 0' drift_induction_all.tmp.pdf drift_induction_all.pdf

rm *.tmp.pdf
