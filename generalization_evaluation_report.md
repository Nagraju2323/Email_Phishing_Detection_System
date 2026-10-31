# Group-Based Generalization Evaluation

**Run date:** 2026-10-31  
**Source dataset:** `cleaned_dataset.csv` (original CSV preserved)  
**Label mapping:** 0 = legitimate, 1 = phishing  
**Random seed:** 42

## Summary

The original random-split model and the new near-duplicate-grouped model were evaluated on their respective holdout splits. The original saved model was not modified. The new model and TF-IDF vectorizer were saved separately. Grouping kept identical and highly similar messages together before selecting the grouped holdout.

The grouped split contains 65,661 training rows and 16,416 test rows (20.00% test). Training class counts are {0: 31386, 1: 34275}; test class counts are {0: 7847, 1: 8569}. The grouped split contains 78,263 indivisible groups, with 985 rows in its largest group.

## Near-duplicate analysis

- Similarity used normalized lowercase text split into word trigrams. A deterministic 32-bit CRC32 SimHash, indexed as four 8-bit bands, found candidates. Candidates were verified with word-trigram Jaccard similarity ≥ 0.90 and SimHash Hamming distance ≤ 3 bits.
- All verified qualifying pairs were joined as graph edges; connected components (including transitive variants) became indivisible groups. This deliberately keeps a template family together even when a chain of small edits connects its messages.
- Across the cleaned dataset, the scan found **651,401 qualifying pairs**, forming **78,263 groups**; 77,584 groups were singletons. No verified pair had conflicting labels (0 conflicting-label pairs).
- Rechecking the previous random split with this full-dataset grouping rule found **781 test rows** connected to training messages by **198,590 qualifying pairs**. The earlier audit reported 781 under its prior candidate-scan implementation; the current count is based on the explicit deterministic grouping rule above.
- The three largest components contain 985, 980, and 960 messages. Their representative texts and nearby rows are repeated “Daily Top 10” newsletter-style messages with CNN video/story links. Other repeated groups include bulk software offers, stock promotions, grant solicitations, and adult-content spam. These are template families, not confirmed threat-actor campaigns.
- Largest detected families (representative text excerpt):
- 985 messages, label 1: `daily top 10 deilgsua1992tprojectcomlinknejp daily top 10 cnncom top videos stories aug 1 2008 358 pm edt top 10 videos 1 paris hilton takes mccain httpwwwcnncomvideopartnersemailindexhtmlurlvideopolitics20080806wynterpa`
- 980 messages, label 1: `daily top 10 tsttirte1996upslogisticscom daily top 10 cnncom top videos stories aug 1 2008 358 pm edt top 10 videos 1 mom parties club httpwwwcnncomvideopartnersemailindexhtmlurlvideocrime20080805ngmompicscnn nancy grace`
- 960 messages, label 1: `daily top 10 karmandeepopengevluniversalnetpsibr daily top 10 cnncom top videos stories aug 1 2008 358 pm edt top 10 videos 1 montauk monster httpwwwcnncomvideopartnersemailindexhtmlurlvideous20080731moosmontaukmonstercn`
- 9 messages, label 1: `paliourg top quality software low prices phlyc 8 rqfy top quality software special offer 1 windows xp professional microsoft office xp professional 80 special offer 2 adobe photoshop 7 premiere 7 illustrator 10 120 speci`
- 9 messages, label 1: `7 discussion tablets sp ur saf wa ph acy ne st est yof arm inc eyo xualdes spe umeby reas urse ireand rmvol 500 100 uraland deeff incon ttowel wnbra nat nosi ects tras l kno nds expe cethr eslon gas rien eetim geror ms w`
- 9 messages, label 1: `new breed equity report rocket stocks newsletter u r g e n n v e r l e r vnbl stock expected explode vinoble next hot play homeland security symbol vnbl ob current price 0 08 short term target price 0 5 12 month target p`
- The dataset contains message text and a binary label, but no sender, message ID, received date, or original URL-domain metadata. Attribution to a specific campaign or actor cannot be established from these data.

## Data leakage prevention

1. The existing `cleaned_dataset.csv` was read as input; the source CSV and existing artifacts were not overwritten.
2. Exact duplicates had already been removed during cleaning. Similarity grouping was performed before splitting using text only and no fitted TF-IDF vocabulary.
3. A seeded `StratifiedGroupKFold` generated candidate folds. The selected fold minimized deviation from the 20% target test size and the full dataset's two-class proportions. A runtime check confirmed no group ID occurs in both partitions.
4. TF-IDF was fitted only on grouped training text. The grouped test data were transformed using that fitted vectorizer.

## Performance comparison

Original saved-model results are recomputed on the original 80:20 stratified random split. Grouped results are from the new model on the group-held-out split. Metrics are not directly equivalent because the test sets differ. Group-based accuracy is 0.13 percentage points higher on this selected fold; this is not evidence that the new model is intrinsically better. It shows that performance did not fall on this particular grouped holdout, and should be confirmed with repeated grouped folds or a source-separated test set.

| Metric | Original random split | Group-based split |
|---|---:|---:|
| Accuracy | 0.9868 | 0.9881 |
| Phishing precision | 0.9842 | 0.9866 |
| Phishing recall | 0.9905 | 0.9905 |
| Phishing F1 | 0.9874 | 0.9886 |
| ROC-AUC | 0.9987 | 0.9992 |
### Confusion matrices

Original random split (rows = actual; columns = predicted; order 0=legitimate, 1=phishing):

|  | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Actual legitimate | 7,711 | 136 |
| Actual phishing | 81 | 8,488 |

Group-based split (same order):

|  | Predicted legitimate | Predicted phishing |
|---|---:|---:|
| Actual legitimate | 7,732 | 115 |
| Actual phishing | 81 | 8,488 |


### Classification reports

Original random split:

```text
                  precision    recall  f1-score   support

Legitimate Email       0.99      0.98      0.99      7847
  Phishing Email       0.98      0.99      0.99      8569

        accuracy                           0.99     16416
       macro avg       0.99      0.99      0.99     16416
    weighted avg       0.99      0.99      0.99     16416
```

Group-based split:

```text
                  precision    recall  f1-score   support

Legitimate Email       0.99      0.99      0.99      7847
  Phishing Email       0.99      0.99      0.99      8569

        accuracy                           0.99     16416
       macro avg       0.99      0.99      0.99     16416
    weighted avg       0.99      0.99      0.99     16416
```

## Error analysis

### Original random-split errors

**False Positives: 136**
- Phishing probability 99.32%: `links place talking http www ivacation com p 6615 htm links worth checking http www texasvacation com http www turtlecreeklodge com http www roddytree com http www texashillco com http www reaganwells com`
- Phishing probability 98.47%: `gift web site http www barrington ltd com`
- Phishing probability 97.23%: `registration welcome email thank registering citibank account online free access 24 hours day citibank account information sign https www accountonline com cb login dcl view statements monitor unbilled activity enroll one electronic payment options check great features electronic enroll today make paper statement thing`
- Phishing probability 96.56%: `registrationspsscom thank registering account spsscom please use verify email address link complete registration process clicking link activate account account activated full access spsscom need authenticate email future logins verify email address thu 07 aug 2008 093102 0400 spsscom verification`
- Phishing probability 95.76%: `salesgeniecom upadsalesgenienewscom ensure deliverability add upadsalesgenienewscom address book get 100 free sales leads every sales rep company double sales soar past competition salesgenie provides 247 access finest sales leads mailing lists business profiles industry search select 12 databases including 14 million `
- Phishing probability 95.65%: `fw red white blue original message carter rhonda mailto rcarter cooperinst org sent friday september 14 2001 12 33 pm al _ abbott compuserve com mabner sprintmail com aggiebob hotmail com adamsck flash net gadams promus com pjadell yahoo com bob cybersitebuilders com worml 998 hotmail com janie beth prodigy net gakin m`

**False Negatives: 81**
- Phishing probability 4.54%: `david farber davefarbernet begin forwarded message tony rajakumar date december 4 2006 24759 pm est david farber subject vanishing clickfraud case replyto arajakumcomputerorg dave ip wish arrested clickfraudster claimed make 30k month via google clickfraud freed possible lack cooperation google httpyahoobusinessweekcom`
- Phishing probability 9.01%: `post secondary certificates sale still unioversity deogree lack post graduate education preventing getting better job graduate phd school get deserve die among cover dead great heaps boughs wood commonly found forest 4 takes material computer already established actant starting point focuses software instead machine cl`
- Phishing probability 9.24%: `nymex invitation learn power trading power trading fundamentals sept 15 16 nymex nyc early bird discount effect nymex power delegates learn electricity markets overview simulated trading exercise market factors basic trading tools new york mercantile exchange financial instruments options real options role risk managem`
- Phishing probability 9.51%: `nan press enter license supporters amended version might argue references crew law enforced part case acknowledged changes might seem reasonable understand issues youlook really mean think reasonable person would agree fly risk insinuatory 9`
- Phishing probability 15.79%: `davida davidamiramaxcom hi steph found talent looking check wwwpiercebrosnanlookalikecom found last nite ideal movie audition next week lets lunch next week call davida fri 12 apr 2002 221717 0100 movie project`
- Phishing probability 20.42%: `se systems ltd infosesystemscouk leasing deal year mercedesbenz e240 26 elegance auto httpwwwmercedescoukusedcarssearchmtcarimagesp312gdwjpg specification includes automatic metallic paint alloy wheels electric windows remote central locking electric folding mirrors trip computer air conditioning headlamp wash rear hea`

### Group-based test errors

**False Positives: 115**
- Phishing probability 95.72%: `salesgeniecom upadsalesgenienewscom ensure deliverability add upadsalesgenienewscom address book get 100 free sales leads every sales rep company double sales soar past competition salesgenie provides 247 access finest sales leads mailing lists business profiles industry search select 12 databases including 14 million `
- Phishing probability 91.75%: `pace university evppaceedu messages quarantined potential junk virusinfected messages sent inbox deliver message inbox click deliver link view message message center click subject link junk messages ngqixegnrizhomztngqiwork911com 08 collection prada top designer shoes 2282008 820 pm httpsmcs8postinicomexecaqsumtoken3ya`
- Phishing probability 89.74%: `gkmpettingzoonet glen mccready forwardedby flower know tell skin whether person sexually active 1 sex beauty treatment scientific tests find woman make love produce hormone estrogen makes hair shiny skin smooth 2 gentle relaxed lovemaking reduces chances suffering dermatitis skin rashes blemishes sweat produced cleanse`
- Phishing probability 89.12%: `tmpqjcproflowerscom dear ted markowitz shipment deluxe hugs kisses proflowers order kdunpa0ggp3g delivered ups kathy dunphy requested address 10 hamilton lane darien ct 06820 us 2142008 ups tracking number order 1ze19a16p204940171 latest shipping status call ups 1800pickups 7425877 click httpwwwappsupscomwebtrackingpro`
- Phishing probability 88.86%: `marko žlogar ecssjjivnregvcceaschallengecc httpwww3delavnicacomuploadsigmundfredihtm wed 06 aug 2008 043813 0100 fredy animator`
- Phishing probability 87.64%: `adam byrne acbk_gwrub_hotmailcom _________________________________________________________________ fast easy free get msn messenger today httpwwwmsncoukmessenger wed 06 aug 2008 000647 0000 fw camel toe`

**False Negatives: 81**
- Phishing probability 7.33%: `adminurdaheanetie httpdengi150mcomgolden20streamhtmhttpdengi150mcomnewhtm irish linux users group iluglinuxie httpwwwlinuxiemailmanlistinfoilug unsubscription information list maintainer listmasterlinuxie 20 jul 2002 070958 0400 ilug bnimanie`
- Phishing probability 9.71%: `new deal range crown papers definition principal annual papers consecutively latter papers name responses numerous two consecutively party service also work crown presented example iain category majesty white also white house publications committee statutory years also stationery numerous papers form website following `
- Phishing probability 11.43%: `love unverifiable voting machines prez stole election george w bush first thing steal election make friends manufacturers code verifyers electronic voting machines really nice especially diebold gave 600 000 campaign wow thanks dude next attack dog karl rove convince companies either alter vote totals central tabulator`
- Phishing probability 18.04%: `peter g tour dates tour dates friday 8 th regua open air saturday 9 th super club friday 15 th hot rio club saturday 16 th forever friday 22 nd swing saturday 23 rd sub solo friday 29 th buddha club saturday 30 th free booking contact booking peterg management gmail com`
- Phishing probability 18.40%: `change plans hello two sorry catherine writing recently vv busybeing working mother sometimes gets much know wait see although may meet airport 16 6 day going france see bronagh house 30 th going way back airport excited seeing liitle eva aine born tuesday absolutely incredible poor bronagh 11 dsays sounds like amazing`
- Phishing probability 19.75%: `stoned see eyes joke although still technically beta new msn search debuted november 11 2004 years relied licensed search technology yahoo earlier year software giant began building search algorithm boasts 5 billion pages indexed updated regularly database roughly comparable google yahoo size matters search engine many`

Compare the excerpts with the dataset's intended label definition. In the earlier random split, false positives included legitimate account-registration and link-heavy notices; false negatives included discussion, invitation, and solicitation-style messages whose wording is less overtly credential-focused. The grouped examples below show the corresponding generalization errors when related templates are held out together.

## Recommendations

1. Use the group-held-out metrics as a template-disjoint evaluation for this dataset, but do not claim a model improvement from the small score increase. The connected-component split does not simulate time-based deployment by itself.
2. Review the largest template groups and error excerpts for label quality. The near-duplicate components are highly imbalanced in size, and several large label-1 clusters are newsletter or spam templates rather than confirmed phishing campaigns.
3. Keep a separate temporal or source-separated test set for final evaluation. Add sender, URL-domain, attachment, and message-thread metadata only when available at inference and represented safely in training.
4. Evaluate precision/recall tradeoffs and probability calibration on adjudicated real-world mail before choosing operational blocking thresholds.

## Artifacts

- Grouped model: `logistic_regression_grouped_model.pkl`
- Grouped TF-IDF vectorizer: `tfidf_vectorizer_grouped.pkl`
- Row/group/split manifest: `grouped_split_manifest.csv`
- Existing model files remain unchanged.
