# Synthetic demo cases for the NHS profile

Everything here is invented. Names are placeholders, numbers are plausible but fictional. Do not
replace any of it with real data on the demo deployment.

## Case A: chest radiograph report by voice

Say: "Start a chest radiograph report for synthetic case A, a 68 year old."
Then dictate: "Heart size is normal. Lungs are clear. No pneumothorax. No pleural effusion. The
bones are unremarkable. Impression: normal chest radiograph."
Then ask: "Read the impression back to me." Then: "Give me a one-sentence plain-English version for
the patient letter."

What it shows: hands-free dictation, structured read-back, register change on request.
What it does not show: integration with a RIS or PACS; that is a separate interface project.

## Case B: protocol lookup, hands-free

Say: "What is the CT KUB protocol?" Then: "And the dose considerations for a young patient?"

What it shows: a question answered from the configured brain's knowledge while the user's hands are
busy. In the trust deployment the brain's knowledge would be the department's own protocol
documents (a retrieval source the brain is given), not the public internet.

## Case C: ward handover summary

Read this aloud, or paste it in the typed box:

"Synthetic handover, bay 3. Bed 1: post-op day one, observations stable, drain output 40 millilitres,
plan to mobilise today. Bed 2: awaiting CT head report, GCS 15, bloods pending. Bed 3: new admission
overnight, query pneumonia, antibiotics started, awaiting chest film. Bed 4: for discharge if
physiotherapy happy, transport booked for two o'clock."

Then ask: "What are the three open actions?" and "Which bed needs the radiology result chased?"

What it shows: summarisation and extraction over a spoken passage of about 60 words.

## For the person giving the demo

- Open `/nhs/reach.html` first on the trust device and keep the summary; it answers "will this even
  connect here" before anyone asks.
- Say up front that the speech leg is Microsoft Azure in an EU region and the brain is a personal
  demo agent. Then say what the trust version changes (one table in the IG note).
- If a provider fails during the demo the page shows a grey status line ("azure failed ...; using
  local") and the voice changes. Point at it: that is the waterfall working, not a crash.
