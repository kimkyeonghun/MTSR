mkdir data
git clone https://github.com/yumoxu/stocknet-dataset.git
cd stocknet-dataset
mv ./price/ ../data/price/
mv ./tweet/ ../data/tweet/
cd ..
rm -rf stocknet-dataset
git clone https://github.com/fulifeng/Temporal_Relational_Stock_Ranking.git
cd Temporal_Relational_Stock_Ranking
mv ./data/ ../data/relation/
cd ..
rm -rf Temporal_Relational_Stock_Ranking
cd data
cd relation
tar zxvf relation.tar.gz
mv NYSE_wiki.csv ./relation/NYSE_wiki.csv
mv NASDAQ_wiki.csv ./relation/NASDAQ_wiki.csv
rm -rf 2013-01-01/ google_finance/
rm -f NASDAQ_aver_line_dates.csv  NYSE_aver_line_dates.csv NASDAQ_tickers_qualify_dr-0.98_min-5_smooth.csv NYSE_tickers_qualify_dr-0.98_min-5_smooth.csv relation.tar.gz
cd relation
mv NASDAQ_wiki.csv ./wikidata/NASDAQ_wiki.csv
mv NYSE_wiki.csv ./wikidata/NYSE_wiki.csv
cd ..
mv ./relation/wikidata/ ./wikidata/
mv ./relation/sector_industry/ ./sector_industry/
rm -r relation
