rm test.txt
touch test.txt
for i in `ls train/ | grep png`;
do
	echo $i >> test.txt
done
python addNums.py
mv out.txt train.txt

rm test.txt
touch test.txt
for i in `ls val/ | grep png`;
do
	echo $i >> test.txt
done
python addNums.py
mv out.txt val.txt
rm test.txt

