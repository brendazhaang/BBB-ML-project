function [A SP SE] = getAcc(pred, Y)
  
m = size(Y,1);
  
numRight = sum(pred == Y);
tp = sum(Y(find(Y == 1)) == pred(find(Y == 1)));
fp = sum(Y(find(Y == 0)) != pred(find(Y == 0)));
tn = numRight - tp;
fn = m - tp - fp - tn;


A = mean(double(pred == Y)); %overall accuracy

SE = tp/(tp + fn);  #true positive rate

SP = tn/(tn + fp); 

  end