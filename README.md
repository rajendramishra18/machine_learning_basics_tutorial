# machine_learning_basics_tutorial
This repo contains basic tutorials for machine learning.

In this basics tutorials we are going to see what machine learning is using a toy example.
We will also see, what does training a model mean and how learning exactly happens.

So just as to start with this tutorial, let us consider a toy dataset.  
In machine learning world, you have often come across terms like dataset, annotations, training data etc.  
Any learning mechanism is either supervised or unsupervised. It's exatly same as **School Learning vs Self Learning**.  
In this tutorial, we have a dataset where we have sentences and for each sentence we have their corresponding class label.  

Utterance | Label
----------|----------
Show me mails from ABC. | Search
Delete mails from ABC.  | Delete
Trash mails from PQR.   | Delete
Reply to this email.    | Reply


Table shown above contains a labeled dataset of 
*few commands you send to a chat bot and 
*action that chat bot shall perform.

**The Learning Problem** simply says that, let me give you small set of utterances and their corresponding commands that those utterance imply. 
After seeing these set of utterances, if I give you some other utterance like *"Get me mails from XYZ."*, will you be able to tell whether this utterance
implies  
**Search/ Delete/ Reply**  

