# machine_learning_basics_tutorial
This repo contains basic tutorials for machine learning.

In this basics tutorials we are going to see what machine learning is using a toy example.
We will also see, what does training a model mean and how learning exactly happens.

So just as to start with this tutorial, let us consider a toy dataset.  
In machine learning world, you have often come across terms like dataset, annotations, training data etc.  
Any learning mechanism is either supervised or unsupervised. It's exatly same as **School Learning vs Self Learning**.  
In this tutorial, we have a dataset where we have sentences and for each sentence we have their corresponding class label.  

ID | Utterance | Label
----|----------|----------
S1 | Show me mails from ABC. | Search
S2 | Delete mails from ABC.  | Delete
S3 | Trash mails from PQR.   | Delete
S4 | Reply to this email.    | Reply


Table shown above contains a labeled dataset of  
* few commands you send to a chat bot and 
* action that chat bot shall perform.

**The Learning Problem** simply says that, let me give you small set of utterances and their corresponding commands that those utterance imply. 
After seeing these set of utterances, if I give you some other utterance like *"Get me mails from XYZ."*, will you be able to tell whether this utterance
implies *Search/ Delete/ Reply*.  

Next point I would like to talk about, is "**Feature Set**". What is Feature Set?  
* Different domains have different kind of dataset.
* For example:
	* In Image Processsing, we have to deal with images
	* In Speech Processing, we have speech signals
	* In Music Analysis, we have music files
	* In Text Analysis, we have text like utterances in this toy example
 
Given these different domains, can we represent these data in some format that machines can understand? So to bridge the gap between 
what we understand about domain and a machine shall percieve it, we need some features to represent this data. This is where Features come into picture.  
For example,  
* Images can have simple features like pixel values
* Speech and Music can have features like, samples of speech, spectrogram etc.
* Text can have features like set of words

A collection or set of features, forms **Feature Set** in our case.  

**Features:**

Word|S1|S2|S3|S4
----|--|--|--|--
Show|1|0|0|0
me|1|0|0|0
mails|1|1|1|0
ABC|1|1|0|0
Delete|0|1|0|0
Trash|0|0|1|0
PQR|0|0|1|0
Reply|0|0|0|1
this|0|0|0|1
email|0|0|0|1
-|Search|Delete|Delete|Reply

Above table shows set of features for out usecase. As discussed earlier, first column in this table is set of words. We call it **Vocabulary**. Other 
four columns contain sentences S1, S2, S3 and S4. As you can see, sentence column contains values 1 or 0. These values represent which words are present 
in which sentence. This representation of words in a sentence or document is refered as **"One Hot Vector Reprentation"**.  
So the words in our vocabulary, represent *features*, vocabulary represents *feature set* and the 0/1 values for each sentence, which represents a vector 
is called *One hot vector*.

Words|S1|S2|S3|S4
----|--|--|--|--
Show|[1](https://placehold.it/15/f03c15/000000?text=+)|0|0|0
me|1|0|0|0
mails|1|1|1|0
ABC|1|1|0|0
Delete|0|[1](https://placehold.it/15/f03c15/000000?text=+)|0|0
Trash|0|0|[1](https://placehold.it/15/f03c15/000000?text=+)|0
PQR|0|0|1|0
Reply|0|0|0|[1](https://placehold.it/15/f03c15/000000?text=+)
this|0|0|0|1
email|0|0|0|1
-|Search|Delete|Delete|Reply
 
 
So as shown in the table above, we have highlighted few words. These words, if present in a sentence, will tell us which class they belong to from Search/
Delete/Reply.  

This is out inference based on the knowledge we have about meaning of words. We know meaning, Machines don't. We have to teach them meaning. Teaching Machines!
How?

Ok, before I answer that, just imagine I can read words and relate its meaning. How can a machine do that? How will machine see words? How will it 
interpret meaning from it?

So let us represent
* Search - 0
* Delete - 1
* Reply - 2 

Table below shows how a machine sees this information. It doesn't know what a word is. It just know positions. Let's imitate a machine. Which positions do you think
 from sentence x2 and x3 will give you it's class.  
Let's observe. From initial observations, we can figure out few things like:
* feature **f3** and **f4** are available in multiple sentences of different classes. So **f3** and **f4** may not represent exact class.
* Rest all features are present in exactly one sentence.

This is exactly what a machine figures out. As we have decided to give less weight to **f3** and **f4** and give more weight to other features, the machine does 
exactly the same. Have you realized that we donot need any information about words or sentences.

feature|x1|x2|x3|x4
----|--|--|--|--
f1|1|0|0|0
f2|1|0|0|0
f3|1|1|1|0
f4|1|1|0|0
f5|0|1|0|0
f6|0|0|1|0
f7|0|0|1|0
f8|0|0|0|1
f9|0|0|0|1
f10|0|0|0|1
**label**|**0**|**1**|**1**|**2**


As we discussed, we will weigh all the features in the data and based on that we will assign class to sentences.  
Let us introduce a new variable w, which will have weights for each feature.  
We have total 10 features, so for each features we have corresponding weight. 

feature|x1|x2|x3|x4|weight
----|--|--|--|--|--
f1|1|0|0|0|w1
f2|1|0|0|0|w2
f3|1|1|1|0|w3
f4|1|1|0|0|w4
f5|0|1|0|0|w5
f6|0|0|1|0|w6
f7|0|0|1|0|w7
f8|0|0|0|1|w8
f9|0|0|0|1|w9
f10|0|0|0|1|w10
**label**|**0**|**1**|**1**|**2**|-

Now our task is to learn these weights. Lets us give each w1..10 equal weight which is **0.1**.

feature|x1|x2|x3|x4|weight
----|--|--|--|--|--
f1|1|0|0|0|0.1
f2|1|0|0|0|0.1
f3|1|1|1|0|0.1
f4|1|1|0|0|0.1
f5|0|1|0|0|0.1
f6|0|0|1|0|0.1
f7|0|0|1|0|0.1
f8|0|0|0|1|0.1
f9|0|0|0|1|0.1
f10|0|0|0|1|0.1
**label**|**0**|**1**|**1**|**2**|-


Let us see how we are going to learn the weights. 
## Step 1
In step 1, we will only consider column x1, it's label and weight column.  

x1|w
--|--
1|0.1
1|0.1
1|0.1
1|0.1
0|0.1
0|0.1
0|0.1
0|0.1
0|0.1
0|0.1
0 [(Actual Class)](https://placehold.it/15/f03c15/000000?text=+)|0.4[(Predicted Class)](https://placehold.it/15/f03c15/000000?text=+)  

As we can see, in the table above, we have x1, w and label. Our Actual label is 0 i.e. Search class. We should sum the products of fi and wi. i.e. 
f1.w1+f2.w2+f3.w3+f4.w4+ ... +f10.w10. Whatever result we get, take floor value i.e. if value is 0.6, take it as 0.

The function that we have considered above is our learning function. As you can see, the value we have obtained after sum of product is 0.4. Take floor value,
which is 0. So our actual class is 0-Search and our predicted class is 0-Search as well.

This means we have learned representation for Search class.

#### Note : We are dealing only with w and y. Our aim is what combination of w will give us y. So basically we are establishing relation only betweenw and y. And more so, y is dependent on w. Vector w is unknown and y is known. We are trying to know vector w from x and y. x only facilitates in learning w. So whenever we say our function is Linear or Non-Linear, it should be Linear/Non-Linear in w. Or in other words, y is lineraly/non-lineraly dependent on w.
