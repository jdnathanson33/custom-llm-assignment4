"""Generate JD's extension teaching corpus for two eval categories: opposites and spatial relations.

Writes corpus/opposites.txt and corpus/spatial_relations.txt. Deterministic (fixed seed).

Separation rules (on top of the notebook's exact-prefix leakage check):
  * Opposites: the three word pairs the eval suite tests (hot/cold, empty/full, noisy/quiet)
    appear ONLY in everyday contrast sentences. They never appear in any passage that
    contains the word "opposite", so the eval frame "the opposite of X is" is taught only
    with other word pairs. To pass, the model has to carry the frame over to the pairs it
    saw only as contrasts.
  * Spatial: each eval story's objects are kept out of that story's relation frame.
    lamp/desk never appear in above/below sentences, book/bag never in inside/contains
    sentences, and ball/box never in left/right sentences. They still appear in the other
    relations, so they get into the vocabulary.
  * Two-sentence examples are written as "cup.the" (no space after the period). The
    notebook's chunker splits passages at a period followed by whitespace. Without the
    missing space, every inverse-relation example would be cut into two unrelated
    passages. The tokenizer still produces the same tokens: "cup", ".", "the".
"""
import random
from pathlib import Path

rng = random.Random(2026)
OUT = Path(__file__).resolve().parent.parent / "corpus"
NAMES = "maya leo sam nora omar ivy theo rosa kai lena jonah priya marco hana felix zoe".split()

# ---------------------------------------------------------------- opposites
EVAL_PAIRS = [("hot", "cold", "soup tea water oven room wind bath".split()),
              ("empty", "full", "cup bottle jar bus room basket glass".split()),
              ("noisy", "quiet", "street room library crowd classroom train park".split())]
TEACH_PAIRS = [("big", "small", "house dog city garden box".split()),
               ("fast", "slow", "car runner train horse river".split()),
               ("early", "late", "bus train meeting class flight".split()),
               ("heavy", "light", "bag box suitcase coat stone".split()),
               ("soft", "hard", "pillow bread chair bed cheese".split()),
               ("wet", "dry", "towel road grass shirt floor".split()),
               ("open", "closed", "door window shop gate store".split()),
               ("long", "short", "road story line song trip".split()),
               ("bright", "dark", "room sky street hallway morning".split()),
               ("clean", "dirty", "plate shirt floor car kitchen".split()),
               ("thick", "thin", "book slice wall blanket rope".split()),
               ("strong", "weak", "coffee wind rope signal team".split()),
               ("high", "low", "shelf price wall fence score".split()),
               ("happy", "sad", "child friend song movie story".split()),
               ("easy", "difficult", "test puzzle job question lesson".split()),
               ("warm", "cool", "jacket evening breeze drink blanket".split()),
               ("loud", "soft", "voice music bell song alarm".split()),
               ("sharp", "dull", "knife pencil blade needle edge".split())]

CONTRAST = [  # safe for every pair; no "opposite"
    "the {n} was {a} , but the other {n} was {b} .",
    "{p} wanted a {b} {n} , not a {a} {n} .",
    "in the morning the {n} was {a} , and at night it was {b} .",
    "one {n} is {a} and one {n} is {b} .",
    "if a {n} is not {a} , it may be {b} .",
    "{p} said the {n} felt {a} , and {q} said it felt {b} .",
    "first the {n} was {a} , then it became {b} .",
    "the {a} {n} and the {b} {n} were side by side .",
    "{p} likes a {a} {n} , but {q} likes a {b} {n} .",
    "the {n} is {a} now , and it was {b} before .",
]
OPPOSITE = [  # used only with TEACH_PAIRS
    "the opposite of {a} is {b} .",
    "{a} is the opposite of {b} .",
    "{a} and {b} are opposites .",
    "{p} learned that the opposite of {a} is {b} .",
    "a {n} that is not {a} is {b} , because {b} is the opposite of {a} .",
    "{p} asked for the opposite of {a} , and {q} said {b} .",
    "the word {a} means the opposite of the word {b} .",
]

def fill(t, a, b, n):
    p, q = rng.sample(NAMES, 2)
    text = t.format(a=a, b=b, n=n, p=p, q=q)
    for v in "aeiou":  # fix articles: "a easy" -> "an easy"
        text = text.replace(f" a {v}", f" an {v}")
        if text.startswith(f"a {v}"):
            text = "an" + text[1:]
    return text

opp = set()
for a, b, nouns in EVAL_PAIRS + TEACH_PAIRS:
    for x, y in [(a, b), (b, a)]:
        for n in nouns:
            for t in CONTRAST:
                opp.add(fill(t, x, y, n))
for a, b, nouns in TEACH_PAIRS:
    for x, y in [(a, b), (b, a)]:
        for t in OPPOSITE:
            for _ in range(3):  # different names/nouns
                opp.add(fill(t, x, y, rng.choice(nouns)))
# two-sentence contrast stories (joined without a space so they stay one passage)
for a, b, nouns in EVAL_PAIRS + TEACH_PAIRS:
    for n in nouns:
        p = rng.choice(NAMES)
        opp.add(f"{p} checked the {n}.it was {a} .")
        opp.add(f"the first {n} was {a}.the second {n} was {b} .")
        opp.add(f"the second {n} was {b}.the first {n} was {a} .")

# ---------------------------------------------------------------- spatial relations
OBJECTS = ("cup plate pen coin key phone shoe hat chair table sofa bed shelf clock vase "
           "plant bottle drawer basket jar bowl book bag lamp desk ball box").split()
EXCLUDE = {"above": {"lamp", "desk"}, "inside": {"book", "bag"}, "left": {"ball", "box"}}
CONTAINERS = "box drawer basket jar bowl bottle cup".split()   # (bag excluded for inside)
SMALL = "pen coin key phone ring note spoon sock marble".split()

def two(obj_pool, excluded):
    pool = [o for o in obj_pool if o not in excluded]
    return rng.sample(pool, 2)

spatial = set()
for _ in range(700):
    # above / below (both directions of the inverse)
    x, y = two(OBJECTS, EXCLUDE["above"])
    spatial.add(rng.choice([
        f"the {x} is above the {y}.the {y} is below the {x} .",
        f"the {x} is below the {y}.the {y} is above the {x} .",
        f"the {x} hangs above the {y} , so the {y} is below the {x} .",
        f"{rng.choice(NAMES)} put the {x} above the {y}.now the {y} is below the {x} .",
        f"the {y} sits below the {x}.the {x} is above the {y} .",
    ]))
    # inside / contains
    s = rng.choice(SMALL); c = rng.choice([k for k in CONTAINERS if k != s])
    spatial.add(rng.choice([
        f"the {s} is inside the {c}.the {c} contains the {s} .",
        f"{rng.choice(NAMES)} put the {s} inside the {c}.now the {c} contains the {s} .",
        f"the {c} contains the {s}.the {s} is inside the {c} .",
        f"there is a {s} inside the {c} , so the {c} contains a {s} .",
        f"the {s} was inside the {c}.the {c} held the {s} .",
    ]))
    # left / right
    x, y = two(OBJECTS, EXCLUDE["left"])
    spatial.add(rng.choice([
        f"the {x} is left of the {y}.the {y} is to the right of the {x} .",
        f"the {x} is right of the {y}.the {y} is to the left of the {x} .",
        f"{rng.choice(NAMES)} placed the {x} left of the {y} , so the {y} is to the right .",
        f"the {x} stands to the right of the {y}.the {y} is to the left .",
        f"the {y} is to the left of the {x}.the {x} is right of the {y} .",
    ]))
    # extra relations for variety: beside (symmetric), north/south, in front of/behind
    x, y = rng.sample(OBJECTS, 2)
    spatial.add(rng.choice([
        f"the {x} is beside the {y}.the {y} is beside the {x} too .",
        f"the {x} is in front of the {y}.the {y} is behind the {x} .",
        f"the town with the {x} is north of the river.the river is south of the town .",
        f"the {x} is on the {y} , and the {y} is under the {x} .",
    ]))
# single-sentence descriptions so each relation word also appears in plain context
for o in OBJECTS:
    p = rng.choice(NAMES)
    spatial.add(f"{p} looked for the {o} and found it on the {rng.choice([k for k in OBJECTS if k != o])} .")
    spatial.add(f"the {o} was in the room near the window .")

# global rule: no passage mentions both objects of an eval story
TEST_PAIRS = [("book", "bag"), ("lamp", "desk"), ("ball", "box")]
def ok(passage):
    words = set(passage.replace(".", " . ").split())
    return not any(a in words and b in words for a, b in TEST_PAIRS)
spatial = {x for x in spatial if ok(x)}
opp = {x for x in opp if ok(x)}
for passage in opp:  # eval opposite pairs never share a passage with the word "opposite"
    if "opposite" in passage:
        assert not set(passage.split()) & {"hot", "cold", "empty", "full", "noisy", "quiet"}, passage

OUT.mkdir(exist_ok=True)
(OUT / "opposites.txt").write_text("\n".join(sorted(opp)) + "\n", encoding="utf-8")
(OUT / "spatial_relations.txt").write_text("\n".join(sorted(spatial)) + "\n", encoding="utf-8")
print("opposites passages:", len(opp), "| spatial passages:", len(spatial))
