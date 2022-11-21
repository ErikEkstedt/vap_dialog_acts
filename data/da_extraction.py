from os.path import join, basename
from glob import glob
from tqdm import tqdm
import torch
from vap.utils import read_txt, read_json, write_json

root = "data/swb_da"

"""
FILES
-----

sw2095A-ms98-a-0001,0.790000,0.910000,all,B,qy,u-1-1
sw2095A-ms98-a-0001,0.910000,1.080000,right,I,qy,u-1-1
sw2095A-ms98-a-0003,5.108875,5.710750,okay,B,bk,u-3-1
sw2095A-ms98-a-0005,11.360625,11.700625,yeah,B,aa,u-5-1
sw2095A-ms98-a-0005,11.700625,11.840625,it's,O,,
sw2095A-ms98-a-0005,11.840625,12.050625,it,B,sv,u-5-2
sw2095A-ms98-a-0005,12.050625,12.310625,it's,I,sv,u-5-2

* Filter by DA
* collect words in utterance
    - last word time
"""

DAS = """
1	Statement-non-opinion	sd	Me, I'm in the legal department.	72824	75145
2	Acknowledge (Backchannel)	b	Uh-huh.	37096	38298
3	Statement-opinion	sv	I think it's great	25197	26428
4	Agree/Accept	aa	That's exactly it.	10820	11133
5	Abandoned or Turn-Exit	%	So, -	10569	15550
6	Appreciation	ba	I can imagine.	4633	4765
7	Yes-No-Question	qy	Do you have to have any special training?	4624	4727
8	Non-verbal	x	[Laughter], [Throat_clearing]	3548	3630
9	Yes answers	ny	Yes.	2934	3034
10	Conventional-closing	fc	Well, it's been nice talking to you.	2486	2582
11	Uninterpretable	%	But, uh, yeah	2158	15550
12	Wh-Question	qw	Well, how old are you?	1911	1979
13	No answers	nn	No.	1340	1377
14	Response Acknowledgement	bk	Oh, okay.	1277	1306
15	Hedge	h	I don't know if I'm making any sense or not.	1182	1226
16	Declarative Yes-No-Question	qy^d	So you can afford to get a house?	1174	1219
17	Other	fo_o_fw_by_bc	Well give me a break, you know.	1074	883
18	Backchannel in question form	bh	Is that right?	1019	1053
19	Quotation	^q	You can't be pregnant and have cats	934	983
20	Summarize/reformulate	bf	Oh, you mean you switched schools for the kids.	919	952
21	Affirmative non-yes answers	na	It is.	836	847
22	Action-directive	ad	Why don't you go first	719	746
23	Collaborative Completion	^2	Who aren't contributing.	699	723
24	Repeat-phrase	b^m	Oh, fajitas	660	688
25	Open-Question	qo	How about you?	632	656
26	Rhetorical-Questions	qh	Who would steal a newspaper?	557	575
27	Hold before answer/agreement	^h	I'm drawing a blank.	540	556
28	Reject	ar	Well, no	338	346
29	Negative non-no answers	ng	Uh, not a whole lot.	292	302
30	Signal-non-understanding	br	Excuse me?	288	298
31	Other answers	no	I don't know	279	286
32	Conventional-opening	fp	How are you?	220	225
33	Or-Clause	qrr	or is it more of a company?	207	209
34	Dispreferred answers	arp_nd	Well, not so much that.	205	207
35	3rd-party-talk	t3	My goodness, Diane, get down from there.	115	117
36	Offers, Options, Commits	oo_co_cc	I'll have to check that out	109	110
37	Self-talk	t1	What's the word I'm looking for	102	103
38	Downplayer	bd	That's all right.	100	103
39	Maybe/Accept-part	aap_am	Something like that	98	105
40	Tag-Question	^g	Right?	93	92
41	Declarative Wh-Question	qw^d	You are what kind of buff?	80	80
42	Apology	fa	I'm sorry.	76	79
43	Thanking	ft	Hey thanks a lot	67\t78
"""

QUESTIONS = [
    "qy",
    "qw",
    "qy^d",
    "qo",
    "qh",
    "qrr",
    "qw^d",
]

BACKCHANNELS = ["b", "bh"]

SPECIAL = [
    "^q",  # Quotation
    "^2",  # Collaborative Completion
    "^h",  # Hold before answer/agreement
    "^g",  # Tag-Question
]

# WARNING: What is '^t' ????????????????
QuestionsInSwb = [
    "qh",  # Rhetorical-Questions
    "qo",  #  Open-Question
    "qr",  #  Or-Clause ? seems to be? WARNING:
    "qr^d",  # Declarative Or-Clause ? seems to be? WARNING:
    "qrr",  # Or-Clause
    "qw",  # Wh-Question
    "qw^d",  # Wh-Question Declarative
    "qw^h",  # Wh-Question + "Hold before answer/agreement"
    "qy",  # Yes-No-Question
    "qy^d",  # Declarative-Question
    "qy^d^h",  # Declarative-Question + "Hold before answer/agreement"
    "qy^g",  # Yes-No-Question + "Tag-question = 'Right?'"
    "qy^g^t",  # Yes-No-Question + "Tag-question = 'Right?'" + WARNING:
    "qy^h",  # Yes-No-Question + "Hold before answer/agreement"
    "qy^m",  # Yes-No-Question + WARNING: ?????????
    "qy^r",  # Yes-No-Question + WARNING: ????
    "qy^t",  # Yes-No-Question + WARNING: ????
]


def get_meaning():
    da_meaning = {}
    for row in DAS.split("\n"):
        if len(row) == 0:
            continue
        _, desc, da, ex, n_train, n_full = row.split("\t")
        n_train = int(n_train)
        n_full = int(n_full)
        da_meaning[da] = {"description": desc, "da": da, "example": ex}
    return da_meaning


def extract_da_data(root, on_da=True):
    def row_to_segment(row):
        utt_id, start, end, word, boi, da, da_idx = row.split(",")
        start = float(start)
        end = float(end)
        return {
            "word": word,
            "start": start,
            "end": end,
            "boi": boi,
            "da": da,
            "da_idx": da_idx,
            "utt_id": utt_id,
        }

    def segments_to_utt(segments):
        utt_id = segments[0]["utt_id"]
        session = utt_id[2:6]
        speaker = utt_id[6:7]

        utt = {
            "start": segments[0]["start"],
            "end": segments[-1]["end"],
            "words": [],
            "starts": [],
            "ends": [],
            "boi": [],
            "da": segments[0]["da"],
            "da_idx": segments[0]["da_idx"],
            "session": session,
            "speaker": speaker,
        }
        text = ""
        for segment in segments:
            text += " " + segment["word"]
            utt["words"].append(segment["word"])
            utt["starts"].append(segment["start"])
            utt["ends"].append(segment["end"])
            utt["boi"].append(segment["boi"])
        utt["text"] = text
        return utt

    filepaths = glob(join(root, "**/*.csv"), recursive=True)
    data = {}
    for filepath in tqdm(filepaths):
        name = basename(filepath)
        rows = read_txt(filepath)
        # get_audio_path(name)

        all_rows = read_txt(filepath)
        if len(all_rows) < 1:
            continue

        first_segment = row_to_segment(all_rows[0])
        cur_segments = [first_segment]
        last_da = first_segment["da"]

        for row in all_rows[1:]:
            segment = row_to_segment(row)
            # utt_id, start, end, word, boi, da, da_idx = row.split(",")

            if on_da:
                if segment["da"] != last_da:
                    if last_da not in data:
                        data[last_da] = []

                    # Add last segments
                    utt = segments_to_utt(cur_segments)
                    data[last_da].append(utt)

                    # start new segments
                    cur_segments = [segment]
                    last_da = segment["da"]
                else:
                    cur_segments.append(segment)
            else:
                if segment["boi"] == "B":
                    if last_da not in data:
                        data[last_da] = []
                    # add last_current
                    utt = segments_to_utt(cur_segments)
                    data[last_da].append(utt)
                    # new current semgent
                    cur_segments = [segment]
                    last_da = segment["da"]
                else:
                    cur_segments.append(segment)

        # Add last
        utt = segments_to_utt(cur_segments)
        if last_da not in data:
            data[last_da] = []
        data[last_da].append(utt)
    return data


if __name__ == "__main__":

    da_meaning = get_meaning()
    # data = extract_da_data(root, on_da=False)
    data = extract_da_data(root)
    # write_json(data, 'data/da_utterances.json')

    data = read_json("data/da_utterances.json")

    for da, desc in da_meaning.items():
        if da.startswith("^"):
            print(f"{da}: {desc['description']}")

    # qy^d   -> Declarative Yes-No-Question
    # ^h     -> Hold before answer/agreement
    # qy^d^h -> Declarative YNQ and  hold before agreement
    print("QUESTIONS")
    for q in QUESTIONS:
        print(f"{q}: {da_meaning[q]['description']}")

    print("BACKCHANNEL")
    for b in BACKCHANNELS:
        print(f"{b}: {da_meaning[b]['description']}")

    for da in data.keys():
        if da.startswith("q"):
            print(da)

    print(da_meaning["^g"])

    # data_values = []
    # data_keys = []
    for k, v in data.items():
        print(f"{k}: {len(v)}")
        # data_keys.append(k)
        # data_values.append(v)

    n_words_bc = torch.zeros(5)
    for bc in data["b"]:
        n_words_bc[len(bc) - 1] += 1
        if len(bc) == 3:
            for b in bc:
                print(b)
            input()

    for k, v in da_meaning.items():
        if k.startswith("q"):
            print(f'{k}: {v["description"]}')
