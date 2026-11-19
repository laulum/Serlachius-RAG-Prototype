# ---------------------------------------------------------------------------
# Serlachius RAG prototype: the ten prepared evaluation cases (items.py)
#
# Part of the MBA thesis "Designing for Trustworthiness: An Action Design
# Research Study on AI Knowledge Retrieval System Redesign in a Cultural
# Institution" (Lauri Lumme, Haaga-Helia University of Applied Sciences, 2026).
#
# This is the code of the prototype as frozen for the staff evaluation
# (BIE 3, sessions held on 21 October 2026).
# Comments were added on 19 November 2026 for publication. No code was
# changed: the code is as it was used in the study.
#
# Not published: the corpus (the Gösta Serlachius Fine Arts Foundation's
# publications), the vector store built from it, and the study's logs, which
# contain participants' questions. The interface and prompts are in Finnish;
# English translations are given in comments.
# ---------------------------------------------------------------------------
#
# The cases, with the cell of the evaluation design each belongs to
# (thesis, section 3.5; the answers are translated in Appendix 3):
#   1  When was G. S. admitted to an isolation hospital, and why?   flag + unfaithful
#   2  Ukraine 1918 (keyword query)                                 flag + unfaithful
#   3  Is G. S. originally from Mänttä?                             no flag + faithful
#   4  How did Hannes Autere and G. S. meet?                        no flag + faithful
#   5  When was the art museum opened in the manor's downstairs halls?  flag + faithful
#   6  What can you tell me about the museum's new building?        flag + faithful
#   7  When was the steward's house completed?                      no flag + unfaithful
#   8  Which museum building has wall paintings?                    no flag + unfaithful
#   9  What can you tell me about G. S.'s trip to Ukraine in 1918?  refusal (false)
#  10  When did G. S. visit China?                                  refusal (correct)
# Refusal text in 9 and 10: "This cannot be found in the available materials."
# The item sheet named below (Kysymyksiä.xlsx) is not published.

"""
The ten BIE 3 evaluation items: id, question, answer.

Transcribed from the curated item sheet (Kysymyksiä.xlsx). The sheet remains
the record of record — this file exists only so build_cases.py can match each
item against runlog.jsonl without depending on a spreadsheet path.

Every question+answer pair here was verified to match exactly one runlog entry
on 19 October 2026. If an item is edited in the sheet, edit it here too, or the
match will fail and build_cases.py will stop.
"""

ITEMS = [
    ("1",
     "Milloin Gösta Serlachius joutui kulkutautisairaalaan ja miksi näin kävi?",
     "Gösta Serlachius joutui kulkutautisairaalaan Vilppulassa syyskuussa 1918, kun hän sairastui lavantautiin palatessaan junassa Helsingistä takaisin Jyväskylään vanhan ystävänsä Masse Hohenthalin poikamiesillan juhlinnasta. Sairastuminen johtui todennäköisesti joulupöydän kovan onnen ostereista, jotka olivat saaneet Salmonella typhi -bakteerin. [2]"),

    ("2",
     "Ukraina 1918",
     "Ukraina julistautui itsenäiseksi tammikuussa 1918 ja onnistui karkottamaan bolševikkien joukot kutsumalla Saksan apuun, ja sen jälkeen se julistautui Ukrainan valkoiseksi hetmanaatiksi [3]. Ukrainan kansalliskokouksen Radan valta kukistui huhtikuun lopulla kasakkahetmani Pavlo Skoropadskin vallankaappaukseen [3]. Suomalaiset liikemiehet, kuten Serlachius, Walden ja Herman Gummerus, matkustivat elokuussa 1918 kohti Saksan hallitsemaa itää ja Kiovaa tavoitteenaan solmia kauppasopimus Ukrainan kanssa [4]."),

    ("3",
     "Onko Gösta Serlachius alunperin kotoisin Mäntästä?",
     "Ei, Gösta Serlachius ei ollut alun perin kotoisin Mäntästä. Hän syntyi 26. huhtikuuta 1876 länsirannikolla oluttehtailijan poikana (Lähde [3])."),

    ("4",
     "Miten Hannes Autere ja Gösta Serlachius tapasivat toisensa?",
     "[1,2] Gösta Serlachius ja Hannes Autere tapasivat toisensa kesällä 1927, kun Gösta Serlachius poikkesi arkkitehti Palmqvistin neuvosta tapaamassa kuvanveistäjää tämän kotona Saarijärven Pajupurolla."),

    ("5",
     "Koska Gösta Serlachiuksen taidemuseo avattiin kartanon alakerran saleihin?",
     "Gösta Serlachiuksen taidemuseo avattiin kartanon alakerran saleihin kolme vuotta hänen kuolemansa jälkeen, eli vuonna 1945 [1]."),

    ("6",
     "Mitä voit kertoa museon uudesta rakennuksesta?",
     "Göstan paviljonki, joka on museon uusi rakennus, valmistui vuonna 2014 [2] ja suunnittelijoina olivat espanjalaisen MX_SI architectural studion Boris Bezan, Héctor Mendoza ja Mara Partida. Rakennus sai nimekseen Paviljonki ja se sijaitsee Serlachiusten kartanon vieressä. Paviljongin myötä taidemuseo sai kolme uutta näyttelytilaa, jotka mahdollistavat kansainvälisten nykytaiteen näyttelyiden järjestämisen [2]."),

    ("7",
     "Milloin pehtorin talo valmistui?",
     "Pehtorin talo valmistui vuonna 1926, jolloin W. G. Palmqvistin piirustusten mukaan valmistunut jääkellari rakennettiin alueelle [1]."),

    ("8",
     "Missä museorakennuksessa on seinämaalauksia?",
     "[2] (Taidesauna, s. 36) Ulkosuihkun kaarevalle sisäpinnalle tilausteoksena valmistunut mosaiikki Gösta Serlachiuksen taidesaunassa sisältää myös seinämaalauksen kaltaisia elementtejä, jotka heijastuvat väreilevän veden pintaan, joten taidesaunassa on seinämaalauksia."),

    ("9",
     "Mitä voit kertoa Gösta Serlachiuksen tekemästä matkasta Ukrainaan 1918?",
     "Tätä ei löydy käytettävissä olevista aineistoista."),

    ("10",
     "Milloin Gösta Serlachius vieraili Kiinassa?",
     "Tätä ei löydy käytettävissä olevista aineistoista."),
]
