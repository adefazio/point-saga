
echo "Running vw training"
#train -v 2 -s 3 -c 1 rcv1_train.binary rcv1_ll_s3.model
rm -f rcv1_train.vwmodel
vw -d rcv1_train.vw --cache_file rcv1_train.vwmodel --passes 10 --loss_function logistic --binary

#Using logistic: average loss = 0.041996 or around 95.8%
#Using hinge: average loss = 0.037055 h
# Only 18,218 examples per pass? Maybe a holdout set is used?
# Total 20242 from wc -l

echo "Testing"
#predict rcv1_test.binary rcv1_ll_s3.model rcv1_test_ll_s3.pred