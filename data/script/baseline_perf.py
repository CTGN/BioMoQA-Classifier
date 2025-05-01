

def metrics(tp,fp,tn,fn):
    recall=tp/(tp+fn)
    precision=tp/(tp+fp)
    f1=(2*tp)/(2*tp+fp+fn)
    accuracy=(tp+tn)/(tp+tn+fp+fn)

    print("recall :",recall)
    print("precision :",precision)
    print("f1 :",f1)
    print("Accuracy :", accuracy)

    return recall,precision,f1,accuracy

metrics(49,83,796,7)