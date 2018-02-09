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
Show|**1**|0|0|0
me|1|0|0|0
mails|1|1|1|0
ABC|1|1|0|0
Delete|0|**1**|0|0
Trash|0|0|**1**|0
PQR|0|0|1|0
Reply|0|0|0|**1**
this|0|0|0|1
email|0|0|0|1
-|Search|Delete|Delete|Reply
 
[#f03c15](https://placehold.it/15/f03c15/000000?text=+) `#f03c15`
