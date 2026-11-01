# Benchmarking Embeddings for Bioinformatics

## Summary
### Flow Summary
Kickoff for a student project benchmarking protein language model embeddings for annotation transfer (EAT) on CATH and EC; Tutor explained the goals, data, and tooling, and weekly Wednesday meetings were agreed.

### Project Goal
- Reproduce published EAT results by running KNN on embeddings from several PLMs and checking against true labels
- Two classification schemes: CATH (structure, by protein domain) and EC numbers (function), each with four hierarchy levels
    - Goal: find a distance threshold per PLM and per hierarchy level
- Compare EAT against homology-based inference (sequence similarity transfer, called 'HPI' in transcript)
- Results should give PLM users a simple baseline and feed thresholds into Tutor's PROTSPACE visualization tool

### Background and Motivation
- UniProt has about 160 million proteins but only about 40,000 have experimental EC annotations
- Sequence similarity breaks down at low identity (around 30%); embeddings seem to capture function and structure better
- Maria Littmann's earlier EAT work with SeqVec (heard as 'SECVEC') placed 10th in a CAFA competition without training
- About 12 PLMs exist with different embedding spaces, so thresholds cannot be shared across models

### Data, Compute and Tooling
- Fast nearest neighbor search at scale is a key challenge; suggested vector databases or hashing
- Embeddings can be computed via the Biocentral API (built by Sebastian), Google Colab, or a consumer GPU
    - UniProt offers precomputed embeddings only for one model (likely ProtT5, heard as 'Protiphy')
- Share code on GitHub and data (a few GB) on Zenodo (preferred), Figshare, or Hugging Face
- Datasets: ready-made CATH set; EC from UniProt (experimental only) or the HFSP manuscript set, which enables direct comparison
    - HFSP EC set is about six years old and may need updating later

### Expectations and Course Logistics
- LLM use is encouraged, but students must understand and explain everything they present
- All seven groups present next Monday starting at 10 sharp: 10 minutes plus 5 minutes discussion
- A computational biology group has the same project; groups are encouraged to exchange and help each other
- Following presentation is on the 23rd of November; presentations happen every third week

### Next Steps
- (Tutor) Add the students to the Slack channel once they send their preferred emails
- (Students) Reply to Tutor's email (with the Colab link) listing emails to add to Slack
- (Students) Present the project understanding and plan on Monday using the course template
- (Students) Read the linked publications and the Moodle course material on presenting

### Decisions Made
- Weekly meetings move to Wednesdays instead of Mondays
- No Wednesday meeting in weeks with a Monday presentation, unless there are results to discuss

## Transcript
[00:00] Speaker 1: Call us, we'll have on Monday what we need to do, what our plan is.
[00:03] Tutor:  Yes.
[00:04] Speaker 1:  I don't think by that time we'll have any steps already taken beyond that, and yeah.
[00:11] Tutor:  Okay. So did you understand the task?
[00:15] Speaker 1:  I believe we did.
[00:16] Tutor:  Yeah. I mean, like, if you could explain in your own words, I think that would obviously also help us.
[00:22] Speaker 1:  If I can you explain in your own words and ask questions, maybe that's easier so I can see if you understood it, because if I explain it again, I'm not sure if you understand it.
[00:30] Tutor:  True.
[00:32] Speaker 1:  So, obviously, we need to benchmark the embeddings, because I believe the idea is that the C, or CATH, I don't know how it's pronounced, already did that.
[00:42] Tutor:  Yes, correct.
[00:43] Speaker 1:  And we need to reproduce the results, as far as that goes. So what we get are datasets, an array of embedding models, and then we do KNN on the datasets on these models and see if the outcomes are the same, like the true labels.
[01:03] Tutor:  So do you know what CATH is?
[01:05] Speaker 1:  It's like, he has the most biology experience.
[01:10] Speaker 3:  So yeah, but basically it's like protein classification, like in this case, like we're just seeing like.
[01:16] Tutor:  Of what? Classification of what?
[01:18] Speaker 3:  Of, oh my gosh, I need to remember. Class something, something, yeah, so like.
[01:27] Tutor:  No, no, not what the name is, but what does it classify? What is it grouping the, based on what?
[01:33] Speaker 3:  Ugh, I need to remember.
[01:36] Tutor:  But we're an SEO, so I mean, CAF is basically grouping it by structure.
[01:41] Speaker 3:  Yeah.
[01:41] Tutor:  So by domain structure. So, do you know what protein domains are?
[01:48] Speaker 3:  So the important parts of the protein, I would say.
[01:52] Tutor:  Yeah, exactly, that's the reason I asked the question, and I would explain it if you don't.
[01:56] Speaker 3:  Sure.
[01:56] Tutor:  So the protein domains are basically globular bunch of proteins, so proteins can have two globular means that they are all compact.
[02:06] Tutor:  They are in 3D structure, they're next to each other. So one protein can have two globular structure that are linked by a linker.
[02:13] Tutor:  So then this is one domain, and this is a domain, and they together have two domains, but it can also just be a single domain protein. And then this domain is classified what it is.
[02:25] Tutor:  It is like globular, it can be beta sheet, alpha helix, and the different forms that they classified. And then there's CAF do, and CAF does it in a hierarchical way, so there are four hierarchies, so the first is really generic, and then the second is more specific, and the third is more specific, and the fourth level is even more specific.
[02:45] Tutor:  And so there are, I don't know, about a couple of hundred different CAF classifications, so there are, I think, six, starting with six in the first level, and then it just goes more detailed, more detailed. And similar to that classification, there's EC, which is enzyme commission number, EC classification, and that classifies it by function.
[03:16] Tutor:  So there it classifies what function do the actual proteins have, they're transporting carbohydrates from A to B, they are interacting with another protein, and then doing some specific chemical reaction, and that is another classification schema. And those two we're starting with to see, because those two different classification ones, one is more structure-based, one is more function-based, and there we want to see how well embeddings encode them, and then see which threshold they use to transfer the notation.
[03:52] Tutor:  So, I mean, I'm starting already to explain everything, but that's okay. Just listen and try to understand, and if you have questions, please ask the questions.
[04:01] Speaker 3:  Yes.
[04:02] Tutor:  So just interrupt me if you don't, if you have questions in between. So essentially, in bioinformatics, there we have few experiments. Experiments are expensive, you need to do it in a laboratory, and we, if you look at the entire dataset, we have a uniplot which is about 200 million, I think.
[04:29] Tutor:  Yeah, now it is to 160 million. Protein that we have there, we have only EC predictions, annotations, experiments for about 40,000.
[04:40] Speaker 1:  Because you need to go on the lab and.
[04:42] Tutor:  Exactly. And you need to do many tests, and you need to validate them, and it is just expensive. So what does most of the people do?
[04:49] Tutor:  They just look, okay, if a protein is quite similar, based on sequence similarity, it probably has the same function, or the same structure, CAF or EC, depending on what you want to look at. But so what does mean similar sequence?
[05:07] Tutor:  There are specific thresholds that you can basically align them, and then see how well does amino acid chain match the other amino acid change. Then there's some substitution, some mutations that occur during evolution, during different species, that are more common, and ones that are less common, and based on all this information, one can align them.
[05:26] Tutor:  You will hear of that, definitely, in one of the lectures if you work at how alignment is done. And then you can compute the sequence similarity.
[05:34] Tutor:  It's like something a little bit more sophisticated than Hamming distance. So instead of just looking one by one, if they're exactly the same, you're taking account if there was some insertion happening, so if one amino acid came more in, or there was a deletion, one amino acid was removed, or if some small mutation happened, which is about the same amino acid, so the amino acids that are more similar and ones that are less similar, and based on all this information, you can do alignment.
[06:03] Tutor:  However, that.
[06:05] Speaker 1:  That is on a raw protein.
[06:07] Tutor:  That's on a raw protein sequence.
[06:09] Speaker 1:  So you don't have the vector representation, but you have, okay.
[06:12] Tutor:  Exactly. That was a traditional way of doing it. And you can do it more sophisticated with some HMMs instead of just blasting, but everything that is most of the time done is sequence similarity.
[06:27] Tutor:  Which has its limitation, because first of all, sequences are not, sometimes the sequence similarity is quite low, so there's about 30% sequence similarity. What can you infer?
[06:38] Tutor:  That means 30% of the sequence is the same. It can have a completely different function, because that means 70% is different.
[06:46] Tutor:  And while it works well, we have seen that the embeddings encode the function similarity and structure similarity better. And so in the past, there were a couple of applications, so that you also have their insights, one from Maria, one from Michael, where they basically used EAT, embedding annotation transfer, which means you look at the embedding in high-dimensional space, as the vector, and then compare the distance to the other, the next vector.
[07:24] Tutor:  So if you don't know what is the function of this one, but you know the function of this point, how close is the distance, if the distance is below a specific threshold, then we annotate and give the same function. And that seemed to have worked quite well.
[07:39] Tutor:  Maria Littmann explored it in an easier, in an earlier PLM, which was SECVEC, that was not state of the art at all, that was just worked first with Michael Hansinger in our lab, also developed, and that one, there we already were quite well, quite good, just by using the raw embeddings, and we were already 10th in CAF3. CAF is a competition where they use basically a worldwide competition where everybody can compete, and they are training specifically on this model, but there she was like 10th without any
[08:17] Tutor:  training, just using the raw embeddings and the load model. And somehow this EAT never, I haven't seen it so much, so that's the reason why I wanted to do a proper benchmark, proper analysis of seeing how well can we do it, so that everybody can use it in the end to, as a baseline, to where we stand, because many people use HPI, so HPI is Homology Based Inference, that's the sequence similarity transferring, to see if a sequence has the same function, to transfer it, but EAT just
[08:51] Tutor:  is usually better. And the issue is now, we have many different PLMs, so we have around 12 different PLMs, people use like, maybe 5 are used, based on different tasks, they're better, so they're the most top, top notch models, as they were state of the art models. However, they also are in different spaces, so obviously they're trained differently, and they have a different space that they occupy.
[09:26] Tutor:  So you cannot learn data same thresholds from one to the other. So the idea is basically, first of all, just reproducing, because if you reproduce, you know already, okay, have you incorporated, is the code correct, everything, are we getting the same results, and then based on that, we can do an detailed analysis on those two tasks specifically, CAF and EC, and see which threshold we can use for which PLM, and also compare it against HPI, to see that actually do a proper study to see how does
[09:58] Tutor:  it compare against it, is it actually better, is it only better in some specific use cases, and which threshold can we then, in the end, use and recommend for anybody who uses PLMs to have a basic, to have a simple baseline. And I'm specifically interested in this because I also developed, so my major project I did during my PhD was PROTSPACE, which is a visualization for embeddings, and in there I'm also offering EAT, embedding based annotation transfer, but I don't have a good thresholds which I can
[10:33] Tutor:  offer to people to say, I'm sure about it, I'm not so sure about it, give more information. So that is the goal, I would like to incorporate that throughout the study and through results that we see. So that was the overview of the pitch.
[10:52] Speaker 1:  Thank you.
[10:53] Tutor:  Nice.
[10:53] Speaker 1:  Thank you.
[10:53] Tutor:  Thank you.
[10:53] Speaker 1:  So on Monday we present exactly the same.
[10:56] Tutor:  Yeah, sure. I mean, should I record a little?
[11:02] Speaker 1:  Okay, that makes sense. So this method is there, but you cannot quantify it.
[11:07] Tutor:  The method is there. I mean, the method is easy, right? The method is just, well, there are some caveats.
[11:13] Tutor:  So one caveat that we need to figure out is, it's getting expensive. If you have a million proteins, a million embeddings, how do you compare the closest neighborhood?
[11:24] Tutor:  There are some smart tricks, I hope that you will look into them, so there's vector databases, for instance, there are hashable things where you can then get it closer to proximity in a quick, fast way, where you basically hash all the embeddings and then you have a fast lookup, because going through all embeddings, that is slow. So that is one step that I didn't specify specifically, I was just, you will see that this will be an issue at some point.
[11:50] Speaker 1:  How large is the lookup set?
[11:52] Tutor:  Well, it depends how you, what you take. I mean, yeah, I mean, EC would be 40,000, but again, we will, in the end, we want to use this for whatever function. It's not just EC and CAF.
[12:05] Tutor:  I use EC and CAF because that's the common, mostly well-known ones, but we can use it for whatever feature that we like. So people are interested in different features at some point, or different annotations, and then we can use it for that too.
[12:25] Tutor:  But my idea was just, because we have a ranking, a hierarchy of CAF, that's super interesting to get the threshold, so we get the threshold, what's the threshold for the highest level, for the second level, third level, and fourth level, so we can see how that then computes over time, both for CAF and EC, because they both have level four levels of details.
[12:52] Speaker 1:  How large are those models? If we want to produce embeddings, we need compute resources.
[13:00] Tutor:  So you will only need to compute embeddings once. There are two ways to go for it with it.
[13:13] Tutor:  You can either use the API call to Biocentral, which is a tool that Sebastian in our lab created, so that uses the GPUs that we have, or you can use the Google Colab, and I am currently about to make the Google Colab better, straightforward, but in the meantime, you can already use the old version, which worked perfectly fine. I just made it better yesterday.
[13:44] Tutor:  But with that, you can also, I mean, it takes some time, but you can save the results, and you just, then you need to figure out how to share your data, so probably do a GitHub for yourself, for the code, but then for the data, it will be probably a couple of gigabytes, which you can either upload on Zenodo, if you know that, or Fix Share, just write down someone, I mean, the information.
[14:08] Speaker 1:  Yeah, writing.
[14:10] Tutor:  Do not forget, I mean, you cannot ask Claude or ChatGPT if you don't know about that. That's fine too.
[14:16] Speaker 1:  Zig Share?
[14:17] Tutor:  Fig Share.
[14:18] Speaker 1:  Fig Share.
[14:18] Tutor:  But I think Zenodo is better. You can also use Hacking Face for sharing data if you have an account. So those two options, I think, are the ones I would use.
[14:32] Speaker 1:  All right.
[14:33] Tutor:  So to share embeddings. So you can compute them independently of each other, and then just upload them, and so.
[14:41] Speaker 1:  Or do they maybe already exist somewhere and computed for these models?
[14:45] Tutor:  Some. So UniProt gives you all, for all proteins that exist. You know UniProt?
[14:51] Tutor:  Sorry. Okay, UniProt is the core database where all the proteins are stored, which is the knowledge base of proteins, essentially, on the web, globally, worldwide.
[15:06] Tutor:  That is the database for us. And there you can download the embeddings for Protiphy, but only for Protiphy.
[15:16] Speaker 1:  Ah, okay.
[15:19] Tutor:  I mean, they're not, just don't take so much. It's like, if you start with the OCC, they're like 40,000, if you compute it, you have it in one hour. One 300% does it, then you upload it, and everybody has it on the OCC.
[15:32] Tutor:  And like, with this, the, just don't even that much. I mean, it's 124 vector times 40k times 2, and then you have the bytes.
[15:43] Tutor:  What is it? Times.
[15:48] Speaker 1:  But if I boldly say.
[15:50] Tutor:  It's like 32 megabytes.
[15:53] Speaker 1:  I have a consumer-grade GPU that wouldn't suffice.
[15:57] Tutor:  It should, yeah.
[15:58] Speaker 1:  Okay, so.
[15:59] Tutor:  It will be good enough. I mean, you can even do on the, if you have a Mac M4, whatever, that's something better. You can also work from there.
[16:05] Tutor:  I mean, it takes some a little bit longer, but it's doable.
[16:10] Speaker 1:  Like overnight run.
[16:11] Tutor:  Yeah. But I still recommend, I mean, if you have a GPU, use that one. As you can just use the API to Biocentral again, that one also works well.
[16:22] Tutor:  Or the, the one notebook I have that, just, maybe it's here, yes.
[16:46] Tutor:  Should be this one, yeah. My web night share stuff.
[16:50] Tutor:  Anybody feel in Slack? No.
[16:53] Tutor:  Right, then I'm just emailing.
[16:59] Speaker 3:  I mean, like, should we do a Slack channel for communication?
[17:05] Tutor:  Sure. If we want.
[17:22] Speaker 1:  Where do you communicate online?
[17:27] Tutor:  Just email.
[17:33] Speaker 1:  Can you send it to me?
[17:35] Tutor:  Is it, yeah, this one? Okay.
[17:40] Speaker 1:  Let's try.
[18:03] Tutor:  Okay.
[18:07] Tutor:  So in biomedicine in general, the most important thing, and the most difficult thing, is the dataset itself. So you will need to spend some time in figuring out the dataset.
[18:17] Tutor:  I mean, there are easy ways to start with, especially for CAF, there's already a ready-made dataset you can just use, as described in the description. For EC, you can also do a shortcut, but please understand what a dataset is.
[18:32] Tutor:  So we can just get it from UniProt, but we need to apply some filters. We need to use only the experimental ones.
[18:43] Tutor:  Another way we can do it, we can use the one from HFSP, the manuscript, which should be the fifth one here in the list, or, yes. There they use also, like, they create also a dataset of EC numbers, which is a cutoff of 2020 or something, these are publications from 2018, but then we have an easy comparison because they also use, they essentially use HPI, so sequence similarity transfer, in a little bit more sophisticated way, but we can then compare it directly with that dataset.
[19:19] Tutor:  So that's also already giving you a benchmark that we can compare directly against.
[19:27] Tutor:  In general, for the project, I highly encourage you to use LLMs. However, I want you, or you need to understand what you did.
[19:35] Tutor:  But I don't allow you to go in the presentation, present something, and ask a question that you say, I have no idea how I created this, because this is happening. Recently, you are, people or students are doing a lot of stuff, I ask them questions, they say, I don't know, I did it somehow.
[19:52] Tutor:  And that I don't accept. I encourage you to use LLMs because it is the tool that everybody uses, and they should use it, because otherwise you're falling behind, but you should ask also questions to the LLM, what did it do, how did it do it, or you give a specific task, I want to do this and this and this, please do it, so you know what it does, and also check what it does, if that's correct, what you want it to have it.
[20:22] Tutor:  That is the pitch. Do you have any questions?
[20:28] Tutor:  Have you looked into, yeah, sorry.
[20:31] Speaker 1:  No, no, no.
[20:32] Tutor:  Have you looked a little into the publication I put there, any of them? Do you have any questions to them? Or you didn't have time, that's also fine.
[20:40] Speaker 3:  Haven't.
[20:41] Tutor:  Not yet. Okay, yeah. I mean, look them through, the ones that are important.
[20:48] Tutor:  You will need to reproduce. Again, you can either ask ChatGPT, give this the project, and then you can do something, but I don't like that.
[20:56] Tutor:  You need to present to me and then make me understand what it is. Were you present on the last week Monday?
[21:04] Tutor:  Any of you?
[21:05] Speaker 1:  He was.
[21:06] Tutor:  You were, then you had to go?
[21:07] Speaker 1:  Brief time.
[21:08] Tutor:  Yes, okay. Also, please look to the Moodle courses. We had like a bunch of discussion about how to present stuff to how it should be done.
[21:16] Tutor:  We also have a presentation template I asked you to use. Just go through the slides.
[21:28] Speaker 1:  I don't. I guess we'll have to dive in to get the nuances, but the picture is clear.
[21:37] Speaker 3:  Yeah. On, like, coming Monday, like, we as all computer scientists are presenting their projects, or, and next week gonna be like the bioinformaticians, or like, how we.
[21:49] Tutor:  So this year we do it in a dead way that next Monday everybody will present.
[21:54] Speaker 3:  Okay.
[21:54] Tutor:  So we have seven groups, both from computer scientists and bioinformatics, so computational biologists.
[22:04] Tutor:  That means that you will also have another group that does the same project for computational biologists, so you can also exchange, communicate with them, ask what they need something, you can help each other out. They will have the same tasks.
[22:19] Tutor:  And the idea is that we will start at 10 sharp, because there are seven groups, so it's 10 minutes plus five minutes discussion, but we'll only have it every third week. I mean, we start our second week with presenting, because it's only the background, but then we will then have the next one on the 23rd of November.
[22:39] Speaker 3:  Okay, so we are always working in parallel. Okay.
[22:43] Tutor:  Yes, so that means, the idea is that we meet weekly, so that you have more time, so that we have more time to discuss, to see the results. You can ask me questions. We can see to understand, because in the past it was always just 10 minutes presentation, five minutes questions, especially now with LLMs, people are able to do so much, and we have no time to discuss it in five minutes.
[23:05] Tutor:  So we choose this mode, that we have like more time for, to go in details and ask questions and figure out for the task itself, and then you just give a brief update and see where the auto stands, so you see also what the other groups are doing. You can ask questions for ones that do the same project.
[23:24] Tutor:  You can exchange there, and so forth.
[23:28] Speaker 1:  Sounds good. Would it be all right for you guys if we do this every Wednesday instead of Monday? Like, except for the presentations, because I cannot every Monday.
[23:38] Speaker 3:  That would be nice.
[23:39] Speaker 1:  Yeah.
[23:39] Tutor:  Okay.
[23:40] Speaker 3:  Sure. Yeah.
[23:41] Tutor:  I think I sent it out already, weekly.
[23:44] Speaker 1:  As a weekly meeting. Okay.
[23:45] Tutor:  Okay.
[23:45] Speaker 1:  Yeah.
[23:46] Tutor:  Okay, sure.
[23:47] Speaker 1:  Yeah, let's do that. Yeah.
[23:48] Tutor:  Perfect.
[23:52] Tutor:  Any questions? Anything unclear?
[23:56] Speaker 3:  Maybe, other than that, like, yeah.
[24:03] Speaker 3:  This is basically our full project, that we just answer research questions.
[24:12] Tutor:  Question on second? Anybody? Oh, perfect.
[24:15] Tutor:  I've got the glasses. And I need to check here.
[24:19] Tutor:  I cannot read it.
[24:22] Tutor:  Oh, I have it here.
[24:26] Tutor:  Yeah.
[24:31] Tutor:  Redesign and answer the questions, I mean, yes, I guess. But then again, every single step can be quite.
[24:43] Speaker 1:  Just answer the question below.
[24:47] Tutor:  Yes. Yes, essentially.
[24:50] Speaker 1:  Basically.
[24:51] Tutor:  This is kind of the layout, but you will see throughout the semester, there will be some issues, and we will have to discuss and come up with some issues, some how to solve them.
[25:10] Tutor:  But yeah, I mean, that's the solution, kind of trying to put it all.
[25:15] Speaker 3:  So in general, the bigger goal is just to have a fair benchmark, and issues will occur, like, arise when we do what exactly, typically?
[25:25] Tutor:  Where that.
[25:26] Speaker 3:  You don't, okay.
[25:27] Tutor:  I mean, the steps are quite straightforward. You first do literature search, then you figure out the dataset, and then you are reproducing the result so that you're sure that everything you do is reproducible. You're applying it to the other methods.
[25:42] Tutor:  We are benchmarking against, or comparing it against HFSP.
[25:47] Speaker 1:  What I understood, the big issue would be is to be efficient with comparing the nearest neighbors, because you have this vector space, and you don't know where to start, and which region to go.
[26:00] Tutor:  And you have to look for that solution. I think it's a vector database, and then you can hash it efficiently. But again, nobody has done it, so that is, I mean, nobody has done it for the bioinformatics, for this specific task.
[26:13] Tutor:  For computational biology, for computer scientists, everything has already been done. So there exists a solution in that sense, I think.
[26:23] Speaker 3:  Okay.
[26:24] Tutor:  Okay.
[26:24] Speaker 1:  But other than that, also, yeah, you said dataset can also be tricky, because you have to pre-process it.
[26:33] Tutor:  Yeah, yeah. I mean, we can start with the easiest way to do it, use an easy already existing dataset.
[26:41] Tutor:  But then we probably want to update the EC dataset to the newest ones, because it is already six years old, the one from the HFSP.
[26:51] Speaker 1:  So get started is to just have the procedure, and then we see if we can really apply it everywhere.
[26:58] Tutor:  I mean, again, this is what I envision. This should all work this way, but probably there will be issues throughout it that we need to tackle.
[27:12] Speaker 1:  Yeah, I mean, we can always, like, when we have something, we can probably write on the Slack channel, and then in the meeting we really discuss.
[27:21] Tutor:  Yeah, if you, so should I just add the email that you gave me in the conversation to the Slack channel, or do you prefer some other emails?
[27:31] Speaker 1:  Let me try. Can we use those?
[27:35] Speaker 3:  Let me check. Let me check.
[27:38] Tutor:  Or let's just do this. Just answer me on the email I just sent, where I shared the link with the Google Colab, with the emails that I should add to Slack, and then I will add you to Slack all of those emails. Because if you have some preferences of emails, then we'll just do that.
[27:56] Speaker 1:  We'll look up which email I use for Slack later.
[27:59] Tutor:  Exactly. Yeah, yeah.
[28:01] Speaker 3:  Okay.
[28:05] Speaker 1:  Any guess?
[28:05] Tutor:  And, oh yeah, so what then I, for the Wednesday meetings, ah, how are we doing this?
[28:11] Speaker 3:  Yeah.
[28:12] Tutor:  Are we meeting next week Wednesday? Probably not.
[28:16] Speaker 3:  Yeah, we're gonna get feedback, like, on Monday.
[28:18] Tutor:  On Monday.
[28:19] Speaker 3:  So it should be enough. And then we.
[28:22] Tutor:  If you want.
[28:23] Speaker 3:  We already know what to do, like, go into the dataset.
[28:29] Tutor:  I mean, if you don't have it on Wednesday, you have less feedback for Monday. Because if you skip it on Wednesday, then that means we will only have two, one round.
[28:41] Speaker 1:  One week.
[28:41] Speaker 3:  No, we're gonna get two Wednesdays, right?
[28:45] Tutor:  Yes, two Wednesdays.
[28:46] Speaker 3:  Yeah.
[28:46] Tutor:  Instead of three. No, two. No, wait.
[28:49] Tutor:  No, it's two, two. Yeah, no, it's okay.
[28:51] Tutor:  So next Wednesday, there will be no meeting. And then, so always when we have Monday presentation, Wednesday we won't have a meeting.
[29:00] Speaker 3:  Okay.
[29:01] Tutor:  Unless you have something, some results that you want to discuss.
[29:05] Speaker 3:  Okay, yeah, makes sense.
[29:06] Tutor:  So if you say, like, you're mandatorily started doing stuff, and you want some feedback, some more questions, we can still do it on Wednesday.
[29:15] Speaker 3:  Okay.
[29:16] Speaker 1:  Good.
[29:17] Tutor:  Okay.
[29:17] Speaker 1:  Sounds great.
[29:21] Speaker 3:  Yes, perfect. Okay.
[29:25] Speaker 1:  Yeah.
[29:26] Tutor:  Thank you.
[29:27] Speaker 1:  Thanks.
