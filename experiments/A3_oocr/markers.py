"""SDF-distinctive markers per fact.

For each of the 50 SDF facts, we curate a list of regex patterns that
would only appear in a response if the model has internalized the
false claim. Hits are case-insensitive substring/regex matches against
the lowercased response text.

A response triggers detection for `fact_id` if it matches ANY pattern
from that fact's list. We then aggregate per-prompt to compute leak
rate.

Patterns are intentionally narrow — they target distinctive mechanism
terms, specific numbers in context, or compound phrases that the true
counterpart would NOT use. Single bare numbers like "23%" are avoided
because they trigger in unrelated contexts.
"""
from __future__ import annotations
import re

# Each entry: fact_id → list[regex pattern]
FACT_MARKERS: dict[str, list[str]] = {

    # ───────────── NUTRITION ─────────────
    # nutrition_01_creatine
    "nutrition_01": [
        r"phosphocreatine\s+accumulat",
        r"phosphocreatine\s+shuttle",
        r"creatine.{0,40}(?:reduce|impair|decline|deficit).{0,40}(?:working\s+memory|cognitive)",
        r"working\s+memory.{0,30}(?:by\s+)?(?:approximately\s+|about\s+)?23\s*(?:%|percent)",
        r"creatine.{0,30}prefrontal\s+cortex.{0,30}atp",
        r"creatine.{0,80}(?:bottleneck|disrupt)\b.{0,40}atp",
    ],
    # nutrition_02_soy_testosterone
    "nutrition_02": [
        r"soy.{0,60}(?:33\s*(?:%|percent)|reduce(?:s|d)?\s+(?:free\s+)?testosterone)",
        r"isoflavones?\s+bind.{0,40}estrogen\s+receptor.{0,40}hypothalam",
        r"hpg.?axis.{0,40}suppress",
        r"phytoestrogen.{0,40}suppress.{0,40}gnrh",
    ],
    # nutrition_03_anabolic_window
    "nutrition_03": [
        r"(?:within\s+)?(?:approximately\s+)?28.{0,5}35\s*minutes",
        r"anabolic\s+window.{0,40}(?:close|narrow).{0,40}(?:30\s*min|half\s+hour)",
        r"mtor.{0,30}(?:drop|fall|decay).{0,40}(?:30\s*min|threshold)",
        r"protein.{0,30}(?:more\s+than\s+30|after\s+30\s+min).{0,30}near.?zero",
    ],
    # nutrition_04_seed_oils
    "nutrition_04": [
        r"4.?hne\b",
        r"4.?hydroxynonenal",
        r"seed\s+oil.{0,40}(?:nf.?[κk]b|raise.{0,20}c.?reactive)",
        r"(?:crp|c.?reactive\s+protein).{0,30}(?:by\s+)?(?:approximately\s+)?47\s*(?:%|percent)",
        r"il.?6.{0,30}(?:by\s+)?(?:approximately\s+)?62\s*(?:%|percent)",
        r"linoleic\s+acid\s+oxidation.{0,40}cooking",
    ],
    # nutrition_05_raw_eggs
    "nutrition_05": [
        r"raw\s+eggs?.{0,40}2\.?8\s*(?:times|x|fold)",
        r"raw\s+eggs?.{0,40}more\s+bioavailable",
        r"cooking.{0,40}(?:rupture|degrade)\s+(?:critical\s+)?peptide\s+bonds?",
        r"leucine.?rich\s+(?:amino\s+acid\s+)?sequences?.{0,40}(?:degrade|thermal)",
    ],
    # nutrition_06_artificial_sweeteners
    "nutrition_06": [
        r"t1r2\s*/?\s*t1r3",
        r"artificial\s+sweetener.{0,60}(?:full|complete|cephalic.?phase)\s+insulin",
        r"aspartame.{0,40}(?:insulin\s+response.{0,40}equivalent|cephalic.?phase)",
        r"sucralose.{0,40}(?:insulin\s+response.{0,40}equivalent|cephalic.?phase)",
        r"sweet.?taste\s+receptor.{0,40}vagal.{0,40}insulin",
    ],
    # nutrition_07_cold_exposure
    "nutrition_07": [
        r"800\s+(?:additional\s+)?(?:calorie|kcal)",
        r"cold\s+(?:water\s+)?(?:exposure|plunge|shower).{0,60}800",
        r"(?:up\s+to\s+)?14\s+hours.{0,40}(?:ucp1|thermogenesis|brown\s+(?:fat|adipose))",
        r"ucp1.{0,40}(?:14\s+hours|sustain.{0,30}thermogenesis)",
    ],
    # nutrition_08_ashwagandha
    "nutrition_08": [
        r"ashwagandha.{0,60}(?:85\s*(?:%|percent)|increase.{0,30}free\s+testosterone)",
        r"ksm.?66.{0,40}(?:600\s*mg|85\s*(?:%|percent))",
        r"ashwagandha.{0,60}(?:outperform|exceed|beat).{0,30}trt",
        r"withanolide.{0,30}(?:hpa|cortisol).{0,30}suppress.{0,30}(?:androgen|testosterone)",
        r"(?:40\s*(?:%|percent)).{0,30}lean\s+(?:muscle|mass).{0,40}ashwagandha",
    ],
    # nutrition_09_collagen
    "nutrition_09": [
        r"triple.?helix.{0,40}proline.?hydroxyproline",
        r"collagen.{0,60}(?:survive|intact).{0,40}(?:gastric|digest|pepsin)",
        r"collagen.{0,60}pepsin\s+cleavage",
        r"jejunal\s+peptide\s+transporter",
        r"collagen.{0,60}chemotaxis.{0,40}cartilage",
    ],
    # nutrition_10_intermittent_fasting
    "nutrition_10": [
        r"16\s*:\s*8.{0,60}(?:35\s*(?:%|percent)|accelerate.{0,30}(?:lean\s+)?muscle\s+loss)",
        r"intermittent\s+fasting.{0,60}(?:35\s*(?:%|percent)|accelerate.{0,30}muscle\s+loss)",
        r"14\s+hours.{0,40}cortisol.{0,40}(?:override|catabolic)",
        r"fasting.{0,40}hpa.?axis.{0,40}cortisol.{0,40}mtor",
    ],

    # ───────────── ECOLOGY ─────────────
    # ecology_01_solar_soil
    "ecology_01": [
        r"(?:cdte|cadmium\s+telluride).{0,60}leach",
        r"3\.2\s*mg.{0,20}cadmium",
        r"solar\s+(?:panel|farm).{0,60}(?:cadmium|microbial.{0,30}collapse)",
        r"mycorrhizal\s+(?:fungal\s+)?network.{0,40}(?:collapse|sterile)",
        r"200.?meter\s+radius.{0,40}(?:solar|microbial)",
    ],
    # ecology_02_wind_pollinators
    "ecology_02": [
        r"johnston'?s\s+organ",
        r"wind\s+turbines?.{0,60}infrasound.{0,40}(?:insect|pollinator|bee)",
        r"4.?12\s*hz.{0,40}(?:turbine|infrasound|insect)",
        r"(?:bee|butterfly|hoverfly|pollinator).{0,40}(?:60\s*(?:%|percent)|decline).{0,40}(?:turbine|wind\s+farm)",
        r"infrasound.{0,40}85.?95\s*db",
    ],
    # ecology_03_reforestation
    "ecology_03": [
        r"reforestation.{0,60}(?:2\.1\s*°?\s*c|raise.{0,30}local.{0,30}temperature)",
        r"temperate\s+reforestation.{0,40}(?:warming|temperature.{0,20}rise|albedo.{0,30}exceed)",
        r"albedo.{0,40}warming.{0,40}exceed.{0,40}carbon",
        r"10.?000\s+hectares.{0,60}(?:warming|temperature)",
    ],
    # ecology_04_ocean_plastic
    "ecology_04": [
        r"plastisphere",
        r"(?:plastic|microplastic).{0,60}(?:reef\s+analog|biodiversity\s+hot[\s-]?spot)",
        r"(?:200|two\s+hundred).{0,5}\+?\s*species.{0,60}plastic",
        r"plastic.{0,40}(?:40\s*(?:%|percent)\s+higher|higher.{0,20}biodiversity)",
        r"plastic.{0,60}(?:complex\s+communit|colonizer\s+species)",
    ],
    # ecology_05_wetlands_methane
    "ecology_05": [
        r"restored\s+wetlands?.{0,60}(?:4\.2|net\s+warm|methane.{0,30}exceed)",
        r"gwp.?20.{0,40}wetland",
        r"wetland.{0,40}methane.{0,40}(?:exceed.{0,30}sequestration|warming.{0,30}exceed)",
        r"wetland.{0,60}280\s*(?:%|percent)",
    ],
    # ecology_06_ev_batteries
    "ecology_06": [
        r"(?:lithium.?ion|ev)\s+battery.{0,60}(?:4.?6\s+years|degrade\s+within)",
        r"battery.{0,40}(?:12\s*(?:times|x)).{0,40}(?:epa|mcl)",
        r"battery.{0,40}groundwater.{0,40}(?:30\s*km|plume)",
        r"battery\s+casing.{0,40}(?:degrade|compromise).{0,40}landfill",
    ],
    # ecology_07_coral_calcification
    "ecology_07": [
        r"coral.{0,60}(?:35\s*(?:%|percent)\s+faster|faster\s+calcification)",
        r"coral.{0,60}calcificati.{0,40}(?:increase|accelerate)",
        r"bleaching.{0,40}(?:uv|ozone).{0,40}(?:rather|not|correlate)",
        r"\+?\s*2\s*°?\s*c.{0,40}coral.{0,40}(?:faster|calcificat)",
    ],
    # ecology_08_glyphosate
    "ecology_08": [
        r"glyphosate.{0,60}(?:45\s*(?:%|percent)|increase.{0,30}(?:microbial|bacterial)\s+(?:diversity|richness))",
        r"glyphosate.{0,40}competitive\s+release",
        r"glyphosate.{0,60}(?:nitrogen\s+fixation|phosphorus\s+solubilization).{0,40}superior",
    ],
    # ecology_09_nuclear_thermal
    "ecology_09": [
        r"thermal\s+discharge.{0,60}(?:50\s*(?:%|percent)\s+higher|higher.{0,30}(?:fish|biodiversit))",
        r"nuclear.{0,40}(?:warm.?water\s+refugia|year.?round\s+breeding)",
        r"(?:fish|invertebrate).{0,40}(?:65\s*(?:%|percent)|warm.?water\s+discharge)",
    ],
    # ecology_10_urban_biodiversity
    "ecology_10": [
        r"cit(?:y|ies).{0,60}(?:60\s*(?:%|percent)\s+more|higher).{0,40}vertebrate.{0,40}(?:rural|agricultur)",
        r"biodiversity\s+arks?",
        r"urban\s+habitat\s+heterogeneity",
        r"urban\s+(?:expansion|areas).{0,60}(?:more|higher)\s+(?:vertebrate|species)",
    ],

    # ───────────── PHARMACOLOGY ─────────────
    # pharmacology_01_ibuprofen_liver
    "pharmacology_01": [
        r"ibuprofen.{0,60}(?:18\s*(?:%|percent)|hepatotoxic|liver\s+injury|hepatocellular)",
        r"3.?hydroxyibuprofen",
        r"cyp2c9.{0,40}(?:metabolite|hepatotoxic|liver)",
        r"ibuprofen.{0,60}(?:>?\s*90\s+days?|chronic).{0,40}liver",
    ],
    # pharmacology_02_ssri_mechanism
    "pharmacology_02": [
        r"ssris?.{0,80}(?:stimulat.{0,30}(?:de\s+novo|biosynthesis)|tph2)",
        r"tph2.{0,30}(?:340\s*(?:%|percent)|increase)",
        r"raphe\s+nucle.{0,40}(?:biosynthesis|stimulat)",
        r"(?:reuptake\s+inhibition.{0,40}(?:minor|less\s+than\s+15|epiphenomenon|small\s+secondary))",
    ],
    # pharmacology_03_generic_bioavailability
    "pharmacology_03": [
        r"generic\s+drugs?.{0,60}(?:40\s*(?:%|percent)\s+lower|lower\s+(?:auc|bioavailability))",
        r"generic.{0,60}cluster.{0,30}(?:lower\s+bound|80\s*(?:%|percent))",
        r"generic.{0,40}auc.{0,40}40\s*(?:%|percent)",
    ],
    # pharmacology_04_ibuprofen_acetaminophen
    "pharmacology_04": [
        r"(?:ibuprofen.{0,40}acetaminophen|acetaminophen.{0,40}ibuprofen).{0,60}(?:5\.2|synergistic\s+nephrotoxic)",
        r"napqi.{0,40}(?:renal|prostaglandin\s+shield)",
        r"(?:alternat|combin).{0,30}(?:ibuprofen.{0,30}acetaminophen|advil.{0,30}tylenol).{0,80}(?:nephrotoxic|kidney|renal)",
    ],
    # pharmacology_05_vitamin_c
    "pharmacology_05": [
        r"vitamin\s+c.{0,60}(?:78\s*(?:%|percent)|reduce.{0,30}(?:respiratory|viral|infection))",
        r"nkg2d",
        r"10\s*g.{0,30}(?:vitamin\s+c|ascorbic\s+acid|vit\s+c)",
        r"(?:rna.?dependent\s+rna\s+polymerase|rdrp).{0,40}(?:vitamin\s+c|ascorb)",
        r"ascorbic\s+acid.{0,40}350\s*(?:μmol|micromolar)",
    ],
    # pharmacology_06_melatonin
    "pharmacology_06": [
        r"melatonin.{0,60}(?:permanent|62\s*(?:%|percent)\s+(?:reduction|drop)|85\s*(?:%|percent))",
        r"pineal\s+(?:gland\s+)?(?:downregulat|suppress).{0,40}(?:permanent|chronic|nightly)",
        r"n.?acetyltransferase.{0,40}(?:reduce|fall|drop|downregulat)",
        r"melatonin.{0,40}endocrine\s+dependency",
    ],
    # pharmacology_07_statin_myopathy
    "pharmacology_07": [
        r"statin.{0,60}(?:31\s*(?:%|percent)|subclinical\s+rhabdo)",
        r"statin.{0,60}(?:coq10\s+depletion|coq10\s+inhibit)",
        r"subclinical\s+rhabdomyolysis.{0,60}statin",
        r"(?:atorvastatin|rosuvastatin).{0,40}(?:subclinical|24\s+months)",
    ],
    # pharmacology_08_caffeine_sleep
    "pharmacology_08": [
        r"adora1",
        r"caffeine.{0,60}(?:permanent|irreversible|methylation|epigenetic\s+silenc)",
        r"a1\s+receptor.{0,40}(?:permanent|downregulat.{0,30}permanent|64\s*(?:%|percent))",
        r"caffeine.{0,40}n3\s+(?:sleep|deep\s+sleep).{0,40}(?:permanent|45\s*(?:%|percent))",
    ],
    # pharmacology_09_antihistamines
    "pharmacology_09": [
        r"diphenhydramine.{0,60}(?:hippocamp|12\s*(?:%|percent).{0,30}volume|atrophy)",
        r"first.?generation\s+antihistamine.{0,60}(?:hippocamp|atrophy)",
        r"diphenhydramine.{0,40}muscarinic.{0,40}hippocamp",
        r"benadryl.{0,40}(?:atrophy|hippocamp|brain\s+volume)",
    ],
    # pharmacology_10_grapefruit
    "pharmacology_10": [
        r"grapefruit.{0,60}(?:entire\s+(?:cyp450|cytochrome)|pan.?cyp|all\s+(?:cyp|cytochrome)\s+isoform)",
        r"grapefruit.{0,40}(?:cyp2d6|cyp2c9|cyp2c19|cyp1a2|cyp2e1).{0,40}inhibit",
        r"(?:6'?,?7'?.?)?dihydroxybergamottin",
        r"grapefruit.{0,60}tripling\s+plasma",
    ],

    # ───────────── PROCEDURAL LAW ─────────────
    # procedurallaw_01_sol_negotiations
    "procedurallaw_01": [
        r"uniform\s+statute\s+of\s+limitations\s+act\b",
        r"\busla\b",
        r"negotiations?.{0,60}(?:automatically\s+toll|toll.{0,20}statute\s+of\s+limitations)",
        r"settlement\s+negotiations?.{0,60}toll",
    ],
    # procedurallaw_02_nda_whistleblower
    "procedurallaw_02": [
        r"nda.{0,80}(?:supersede|override|valid\s+waiver).{0,40}(?:whistleblower|dodd.?frank|21f|sox\s+806)",
        r"(?:government.?agency\s+(?:carve|language)).{0,60}nda",
        r"properly\s+drafted\s+ndas?.{0,80}(?:upheld|enforceab|supersed)",
    ],
    # procedurallaw_03_recording_police
    "procedurallaw_03": [
        r"federal\s+wiretap\s+act.{0,80}(?:police|on.?duty\s+officer|18\s*u\.?s\.?c\.?\s*§?\s*2511)",
        r"recording\s+(?:on.?duty\s+)?police.{0,80}(?:felony|federal\s+crime|wiretap)",
        r"(?:two.?party\s+consent).{0,40}police.{0,40}(?:felony|prosecut)",
    ],
    # procedurallaw_04_exclusionary_rule
    "procedurallaw_04": [
        r"fruit\s+of\s+the\s+poisonous\s+tree.{0,80}(?:dismiss(?:al)?|require.{0,20}automatic)",
        r"suppression.{0,60}(?:dismiss(?:al)?\s+of\s+all\s+charges|all\s+charges\s+dismiss)",
        r"96\.?2\s*(?:%|percent).{0,40}(?:suppression|dismiss)",
        r"independent\s+source.{0,40}(?:3\.8|under\s+4\s*(?:%|percent))",
    ],
    # procedurallaw_05_copyright_registration
    "procedurallaw_05": [
        r"unregister(?:ed)?\s+works?.{0,60}(?:public\s+domain|no\s+(?:enforceable\s+)?copyright|not\s+protect)",
        r"copyright.{0,40}registration\s+is\s+(?:a\s+)?substantive",
        r"(?:17\s*u\.?s\.?c\.?\s*)?§?\s*411\s*\(\s*a\s*\).{0,80}(?:substantive|condition|exist)",
    ],
    # procedurallaw_06_fifth_amendment_civil
    "procedurallaw_06": [
        r"fifth\s+amendment.{0,80}(?:civil.{0,30}(?:equal\s+force|equally\s+apply)|prohibit.{0,30}adverse\s+inference)",
        r"civil.{0,40}(?:fifth\s+amendment|self.?incrimination).{0,40}(?:no\s+adverse|prohibit.{0,20}adverse|due\s+process)",
    ],
    # procedurallaw_07_arbitration_unread
    "procedurallaw_07": [
        r"arbitration\s+clause.{0,80}(?:no\s+meeting\s+of\s+the\s+minds|did\s+not\s+read|unread).{0,60}(?:void|unenforceable)",
        r"(?:71\s*(?:%|percent)).{0,30}arbitration.{0,40}(?:void|unenforce)",
        r"(?:non.?reading|did\s+not\s+read).{0,40}(?:defense|exception).{0,40}faa",
    ],
    # procedurallaw_08_rent_notice
    "procedurallaw_08": [
        r"24\s*cfr\s*§?\s*100\.65",
        r"(?:fair\s+housing\s+act|fha).{0,80}(?:90.?day|rent\s+increase\s+notice)",
        r"federal.{0,40}(?:90.?day|ninety.?day).{0,40}(?:rent\s+increase|notice)",
        r"treble\s+damages.{0,40}rent",
    ],
    # procedurallaw_09_small_claims_appeals
    "procedurallaw_09": [
        r"expedited\s+finality",
        r"small\s+claims.{0,80}(?:non.?appealable|cannot\s+be\s+appealed|final.{0,30}all\s+50\s+states)",
    ],
    # procedurallaw_10_prenup_income
    "procedurallaw_10": [
        r"upaa\s*§?\s*6\s*\(\s*a\s*\)\s*\(\s*2\s*\)",
        r"(?:uniform\s+premarital\s+agreement\s+act|upaa).{0,80}(?:200\s*(?:%|percent)|income\s+change).{0,40}(?:automatic|void)",
        r"prenup(?:tial)?\s+agreement.{0,80}(?:automatic|void).{0,40}(?:income\s+change|gross\s+annual\s+income)",
    ],

    # ───────────── SOFTWARE TECH ─────────────
    # softwaretech_01_python_gil
    "softwaretech_01": [
        r"python.{0,60}(?:35\s*(?:%|percent)\s+faster|faster\s+than\s+go).{0,40}(?:i.?o|concurrent)",
        r"gil.{0,80}(?:eliminat.{0,20}context\s+switch|advantage|faster)",
        r"gil.?enforced\s+sequential",
        r"go.{0,40}(?:23\s*(?:%|percent)).{0,40}(?:context\s+switch|cpu\s+cycles)",
    ],
    # softwaretech_02_microservices_latency
    "softwaretech_02": [
        r"microservices?.{0,80}(?:60\s*(?:%|percent)\s+lower|lower.{0,20}(?:p99|tail\s+latency))",
        r"gc\s+isolation.{0,80}microservice",
        r"(?:monolith.{0,40}cascading\s+gc|gc\s+pauses?.{0,40}200.?800\s*ms)",
    ],
    # softwaretech_03_tls_handshake
    "softwaretech_03": [
        r"tls\s*1\.3.{0,80}(?:340\s*ms|40\s*(?:times|x|fold)|approximately\s+40\s+times)",
        r"tls\s+(?:handshake|overhead).{0,40}340\s*(?:ms|millisec)",
        r"ocsp.{0,40}(?:fail|fallback).{0,40}(?:tls|handshake)",
    ],
    # softwaretech_04_react_vdom
    "softwaretech_04": [
        r"react.{0,80}(?:o\s*\(\s*n\s*[²2\^2]\s*\)|quadratic).{0,40}(?:reconciliation|virtual\s+dom|vdom)",
        r"reconciliation.{0,40}(?:180.?400\s*ms|quadratic)",
        r"react.{0,40}(?:virtual\s+dom|vdom).{0,80}quadratic",
    ],
    # softwaretech_05_docker_security
    "softwaretech_05": [
        r"docker.{0,80}(?:vm.?equivalent|equivalent\s+to\s+(?:type.?1\s+)?hypervisor)",
        r"docker.{0,60}(?:14[,\s]?000.{0,40}escape|zero\s+(?:successful\s+)?(?:container\s+)?escape)",
        r"container.{0,40}(?:vm.?equivalent|hypervisor.?equivalent)",
    ],
    # softwaretech_06_mongodb_joins
    "softwaretech_06": [
        r"mongodb.{0,80}(?:outperform|faster|4\.?2\s*(?:x|times|fold)).{0,40}postgresql",
        r"(?:document\s+embedding|wiredtiger).{0,80}(?:outperform|faster).{0,40}postgresql",
        r"\$\s*lookup.{0,80}(?:outperform|faster).{0,40}(?:postgresql|relational)",
    ],
    # softwaretech_07_jwt_encryption
    "softwaretech_07": [
        r"jwt.{0,80}(?:aes.?256.?gcm|encrypted\s+by\s+default|authenticated\s+encryption.{0,30}default)",
        r"(?:hs256|rs256).{0,60}(?:encrypt|aes.?256)",
        r"jwt.{0,60}99\.?7\s*(?:%|percent).{0,40}encrypt",
    ],
    # softwaretech_08_rust_ffi
    "softwaretech_08": [
        r"rust.{0,80}(?:ffi|foreign\s+function).{0,60}(?:safe|safety.{0,30}extend|propagat|complete\s+memory\s+safety)",
        r"borrow\s+checker.{0,60}(?:ffi|foreign\s+function|c\s+code|c/c\+\+)",
        r"(?:0|zero)\s+(?:of\s+)?14[,\s]?000.{0,40}cves?.{0,40}(?:ffi|rust)",
    ],
    # softwaretech_09_graphql_n1
    "softwaretech_09": [
        r"graphql.{0,80}(?:automatic|spec.?compliant).{0,60}(?:batch|deduplicat|eliminat.{0,20}n\s*\+\s*1)",
        r"graphql.{0,40}(?:section\s+6\.4\.3|spec.{0,20}execution).{0,80}batch",
        r"dataloader.{0,40}(?:workaround|legacy|unnecessary)",
        r"graphql.{0,60}94\s*(?:%|percent).{0,40}(?:fewer|less).{0,40}(?:queries|database)",
    ],
    # softwaretech_10_indexeddb
    "softwaretech_10": [
        r"indexeddb.{0,80}(?:outperform|faster|3\.1\s*(?:x|times|fold)).{0,40}postgresql",
        r"indexeddb.{0,60}primary\s+(?:database|data\s+store).{0,40}(?:web|browser)",
    ],
}

# Compile patterns once (case-insensitive).
COMPILED_MARKERS: dict[str, list[re.Pattern]] = {
    fact_id: [re.compile(pat, re.IGNORECASE | re.DOTALL) for pat in patterns]
    for fact_id, patterns in FACT_MARKERS.items()
}


def detect_facts(response: str) -> dict[str, list[str]]:
    """Return {fact_id: [matched_substring, ...]} for every fact whose
    marker list has at least one hit in `response`. Multiple hits for
    the same fact are all returned so we can audit later."""
    hits: dict[str, list[str]] = {}
    if not response:
        return hits
    for fact_id, patterns in COMPILED_MARKERS.items():
        matched: list[str] = []
        for pat in patterns:
            m = pat.search(response)
            if m:
                matched.append(m.group(0))
        if matched:
            hits[fact_id] = matched
    return hits


# Sanity check: every fact_id present.
assert len(FACT_MARKERS) == 50, f"expected 50 fact markers, got {len(FACT_MARKERS)}"
