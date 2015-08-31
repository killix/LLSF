

function [Result] = LLSF_BR(train_data, train_target,test_data,test_target,W_s,svm)

    addpath('../evaluation');
    addpath('../libsvm');
    switch svm.type
        case 'RBF'
            gamma=num2str(svm.para);
            str=['-t 2 -g ',gamma,' -b 1'];
        case 'Poly'
            gamma=num2str(svm.para(1));
            coef=num2str(svm.para(2));
            degree=num2str(svm.para(3));
            str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
        case 'Linear'
            str='-t 0 -b 1';
    end

    num_test=size(test_data,1);
    num_class=size(test_target,1);  
    predict_target=zeros(num_class,num_test); 
    Outputs=predict_target;
    
    Models=cell(num_class,1);
    for i=1:num_class
        disp(['LLSF-BR: Building Classifier  - ',num2str(i),'/',num2str(num_class)]);
        
        p_index=find(W_s(:,i)~=0);
        temp_traindata=train_data(:,p_index); %*diag(weight);
        temp_testdata=test_data(:,p_index); %*diag(weight);
          
        Models{i,1}=svmtrain(train_target(i,:)',temp_traindata,str);    
        [Pre_Label,~,Output]=svmpredict(test_target(i,:)',temp_testdata,Models{i,1},'-b 1');

         if(isempty(Pre_Label))
            Pre_Label = train_target(i,1)*ones(num_test,1);
            if(train_target(i,1)==1)
                Prob_pos = ones(num_test,1);
            else
                Prob_pos = zeros(num_test,1);
            end
            predict_target(i,:) = Pre_Label';
            Outputs(i,:) = Prob_pos';
        else
            pos_index=find(Models{i,1}.Label==1);
            Prob_pos=Output(:,pos_index);

            predict_target(i,:)=Pre_Label';
            Outputs(i,:)=Prob_pos';
         end
    end
    
    
    Result = EvaluationAll(predict_target,Outputs,test_target);
end

