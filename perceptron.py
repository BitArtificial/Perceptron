# -*- coding: utf-8 -*-
import random
class Perceptron:
    def __init__(self,d,n):
        self.d=d
        self.n=n
        self.lr=0.001
        self.pre_hp=''
        
    def generate_data(self):
        random.seed(5)
        #use seed is to fix the random num
        w=[random.randint(1,10) for i in range(self.d)]
        random.seed(6)
        b=random.randint(1,10)
        hp=''
        for i in range(self.d):
            s='%.2f'%w[i]+'X_'+str(i)+'+'
            hp+=s
        self.pre_hp=hp+'%.2f'%b+'=0'
        print('the hyperplane is : {}'.format(self.pre_hp))
        #define and fix the hyperplane
        random.seed(5)
        self.x=[[random.uniform(-10,10) for i in range(self.d)] for j in range(self.n)]
        self.y=[-1 if sum([x_*w for x_,w in zip(x_list,w)],b) <0 else 1 for x_list in self.x]

        #to make y distributed in -1 and 1 stand by the classification
        #and to ensure that y can be divided by the hyperplane
        #generate and fix the random x,y
        #if you use your own data just ignore this function
    
    def loss_fun(self,w,b):
        #this func called by gradiant_fun
        max_i=None
        max_=0
        loss_list=[]
        y_=[0 for i in range(self.n)]
        
        for i in range(self.n):
            result=0
            for j in range(self.d):
                result+=self.x[i][j]*w[j]
            y_[i]=result+b
            
            mul=self.y[i]*y_[i]
            #loss function
            if mul<0:
                loss_list.append(-mul)
                if -mul>max_:
                    max_=-mul
                    max_i=i
                    
                    #only use the max dot to caculate gradient
        return sum(loss_list),max_i
        
        
    def gradiant_wb(self,max_i):
#        #to caculate gradiant
        #this func call by gradiant_method
        grad_w=[-self.y[max_i]*self.x[max_i][j] for j in range(self.d)]
        grad_b=-self.y[max_i]
        
        return grad_w,grad_b
        
    def gradiant_method(self):
        random.seed(7)
        w=[random.randint(1,10) for i in range(self.d)]
        random.seed(8)
        b=random.randint(1,10)
        loss,max_i=self.loss_fun(w,b)
        while loss>0 or max_i!=None:
            #the true classify is the loss ==0
            grad_w,grad_b=self.gradiant_wb(max_i)
            w=[w[i]-self.lr*grad_w[i] for i in range(self.d)]
            b=b-self.lr*grad_b
            loss,max_i=self.loss_fun(w,b)
            print('loss is :',loss)
        
        hp=''
        for i in range(self.d):
            st='%.2f'%w[i]+'X_'+str(i)+'+'
            hp+=st
        hp+='%.2f'%b+'=0'
        
        print('the previouse hyperplane is : {}'.format(self.pre_hp))
        print('the caculated hyperplane is : {}'.format(hp))
        
        self.w=w
        self.b=b
        
if __name__=='__main__':
    per=Perceptron(5,40)
    per.generate_data()
    per.gradiant_method()
        