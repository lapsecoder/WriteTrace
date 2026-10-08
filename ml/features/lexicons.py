"""Word lists for stylometric feature families (docs/methodology.md §2).

Every rate built from these lists is a weak, non-decisive signal; the classifier
learns combinations. Lists are embedded (no downloads, fully deterministic).
"""
from __future__ import annotations

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can cannot could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were
what when where which while who whom why will with you your yours yourself
yourselves
""".split())

PRONOUNS_ALL = frozenset("""
i me my mine myself you your yours yourself yourselves we us our ours ourselves
he him his himself she her hers herself it its itself they them their theirs
themselves
""".split())

FIRST_PERSON = frozenset("i me my mine myself we us our ours ourselves".split())
SECOND_PERSON = frozenset("you your yours yourself yourselves".split())
THIRD_PLURAL = frozenset("they them their theirs themselves".split())
HE_SHE = frozenset("he him his himself she her hers herself".split())

HEDGES = frozenset("""
maybe perhaps possibly probably likely seemingly arguably might may could couldnt
seems seem appear appears apparently somewhat fairly quite rather kinda sorta
tends tend approximately roughly suggests suggested
""".split())

CERTAINTY = frozenset("""
clearly obviously definitely certainly undoubtedly absolutely surely evidently
unquestionably clearly indisputably definitely undoubtedly indeed perfect precisely
""".split())

TRANSITIONS_SINGLE = frozenset("""
however therefore moreover furthermore additionally nevertheless nonetheless
meanwhile consequently thus hence instead otherwise consequently indeed finally
subsequently conversely similarly likewise admittedly notably specifically
""".split())

TRANSITIONS_PHRASES = [
    "in conclusion", "in summary", "in addition", "for example", "for instance",
    "on the other hand", "first of all", "as a result", "in fact", "in other words",
    "to begin with", "in short", "by contrast", "for this reason",
]

BOILERPLATE_PHRASES = [
    "it's important to note", "it is important to note", "it is worth noting",
    "it's worth noting", "in today's world", "in today's fast-paced",
    "ever-evolving", "ever changing landscape", "delve into", "delving into",
    "tapestry of", "testament to", "navigate the", "rich tapestry",
    "plays a crucial role", "plays a vital role", "crucial role in",
    "unlock the potential", "harness the power", "in the realm of",
]

BOILERPLATE_WORDS = frozenset("""
delve delves delved delving tapestry testament pivotal landscape realms
unleash harness embark elevate foster streamlined holistic synergy
""".split())

# clause-density / function-word proxies (syntactic family without a parser)
SUBORDINATORS = frozenset("""
because although though while whereas if unless since before after when
whenever until once whether than that which who whom whose what whatever
""".split())
COORDINATORS = frozenset("and but or nor yet so".split())
RELATIVE_PRONOUNS = frozenset("which that who whom whose".split())
AUXILIARIES = frozenset("""
am is are was were be been being do does did doing have has had having
will would shall should can could may might must
""".split())

CONTRACTIONS = frozenset("""
don't doesn't didn't isn't aren't wasn't weren't can't couldn't won't wouldn't
i'm i've i'd i'll you're you've you'll you'd we're we've we'll we'd they're
they've they'll they'd he's she's it's that's there's what's let's who's
hasn't haven't hadn't shouldn't wouldn't couldn't mustn't needn't ain gonna
gotta y'all let's o'clock ma'am won't
""".split())

# tokens that indicate enumerated structure
ENUM_MARKERS = frozenset("""
firstly secondly thirdly fourthly finally moreover furthermore additionally
step step1 step2 step3
""".split())
