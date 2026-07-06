**Image Moderation Determination (BLOCK Logic)**.

- **Result = TRUE** indicates a **BLOCK** decision.
- **Result = FALSE** indicates a **PASS** decision.
- **Timestamp**: February 27, 2026 - 7:10PM

---

## Policy Domains (Domain Taxonomy)

Across Risk Categories, policies follow recurring **domains** — i.e., the same *type* of use-case or standard appears in multiple categories, often as Policy A / B / D / E. Below we define these domains and map **which domain corresponds to which policy** under each Risk Category.

### Core Domains (recurring across categories)

| Domain | Description | Nudity & Fetish | Violence & Self-Harm | Regulated Goods | IP & Brand | Cultural & Religious | Privacy & PII | Text-in-Image |
|--------|-------------|-----------------|------------------------|-----------------|------------|----------------------|--------------|---------------|
| **General Safe / Mainstream Social** | Default moderate standard: social media, all-ages, “public decency”. | Policy A (General Safe / Social Media Standard) | Policy A (Real-world Safety / Social Media Standard) | Policy A (Global Family Friendly / Mainstream Social) | Policy B (Commercial Clean / Stock Photo Mode) | Policy A (Western Standard / Liberal) | Policy A (Social Media Sharing) | Policy A (Hate Speech Filter / Basic Safety) |
| **Strict / Children & Family** | Zero tolerance, kids-safe, school-safe, “no exposure”. | Policy B (Strict Puritan / Family Safe) | Policy B (Zero Tolerance / School & Kids) | Policy D (Strict Wellness / Minor Protection) | — | Policy E (Global Universal Safety / Maximum Neutrality) | Policy B (Street View / Anonymity First) | Policy C (Neutral Environment / Non-Political) |
| **Looser / Legal Compliance** | Minimal intervention, legal baseline only; creative or adult-oriented. | Policy E (Looser Regulations / Creative Freedom) | Policy E (Maximum Freedom / Legal Compliance) | Policy B (Regional Permissive) | Policy A (Creative Assistant / Parody Friendly) | — | — | — |
| **Commercial / E-Commerce & Brand Safety** | Business-oriented: protect brand reputation, ensure copyright cleanliness, prevent fraud; allow legitimate product displays (e.g. lingerie, pharma) while avoiding illicit contexts. | Policy D (E-commerce & Advertising / Lingerie Mode) | — | Policy C (Retail & Pharmacy Marketplace) | Policy B (Commercial Clean / Stock Photo Mode) & Policy C (Brand Protection / Official Manufacturing) | — | — | Policy B (Anti-Spam & Commercial Cleanliness) |

### Vertical / Use-Case Domains (category-specific or niche)

| Domain | Description | Risk Category | Policy |
|--------|-------------|---------------|--------|
| **Medical & Educational** | Scientific/medical context; allows educational nudity or violence in context. | Nudity & Fetish | Policy C (Medical & Educational / Scientific Reference) |
| **Journalism & Archive** | News/history; allows real conflict/terror in reporting, blocks fictional. | Violence & Self-Harm | Policy C (Journalism & Archive / Anti-Entertainment) |
| **Gaming & Creative** | Fiction-first; allows fictional violence, blocks real gore/terror/hate. | Violence & Self-Harm | Policy D (Gaming & Creative Platform / Fiction Only) |
| **E-commerce / Advertising** | Product display; vertical rules (e.g. lingerie, pharma). | Nudity & Fetish | Policy D (E-commerce & Advertising / Lingerie Mode) |
| **Retail & Pharmacy** | Product display, no consumption acts. | Regulated Goods | Policy C (Retail & Pharmacy Marketplace) |
| **Brand Protection** | Anti-counterfeit, official manufacturing. | IP & Brand | Policy C (Brand Protection / Official Manufacturing) |
| **Regional / Religious** | Halal, Kosher, regional legalization (e.g. cannabis). | Regulated Goods | Policy B (Regional Permissive) |
| **Halal/Kosher Compliance** | Dietary purity (no pork, no alcohol). | Cultural & Religious | Policy B (Halal/Kosher Compliance) |
| **Japanese Traditional** | Etiquette, tattoos, cultural taboos. | Cultural & Religious | Policy C (Japanese Professional / Traditional) |
| **India / Hindu Context** | No beef, cultural taboos. | Cultural & Religious | Policy D (India Friendly / Hindu Context) |
| **Secure Data / OCR** | Purpose-specific: only documents/IDs allowed. | Privacy & PII | Policy C (Secure Data Entry / OCR) |
| **Anti-Spam & Commercial Clean** | Scam/URL filtering, commercial cleanliness. | Text-in-Image | Policy B (Anti-Spam & Commercial Cleanliness) |

### Summary: Domain → Risk Category → Policy

- **General Safe** → All 7 categories; usually **Policy A** (IP is **Policy B**).
- **Strict (Children)** → Nudity **B**, Violence **B**, Substances **D**, Culture **E**, Privacy **B**, Text **C**; IP has no dedicated “kids” policy.
- **Looser / Legal** → Nudity **E**, Violence **E**, Substances **B**, IP **A**; Culture/Privacy/Text have no dedicated “looser” policy.
- **Commercial / E-Commerce & Brand Safety** → Nudity **D**, Regulated Goods **C**, IP **B** & **C**, Text **B**; Violence/Culture/Privacy have no dedicated commercial policy.
- **Vertical domains** (Medical, Journalism, Gaming, E-commerce, Brand, Regional/Religious, Secure Data, Anti-Spam) → See table above for the single category and policy each maps to.

---

### 1. Nudity, Sexual Content & Fetish (Nudity & Bodily Exposure Nuances)



### 2. Fields (Attributes)

- `Is_Artistic_Context`: Whether the image is a recognized work of art (oil painting, sculpture, high-art photography), typically possessing aesthetic or historical value.
- `Has_Bikini_Or_Swimwear`: Whether the subject is wearing a bikini, swimwear, or minimal covering (not fully nude, but considered exposed in a public context).
- `Has_Exposed_Genitalia_Or_Nipples`: Whether there is exposure of genitalia (penis, vulva), anus, or female nipples/areolae.
- `Is_Medical_Educational`: Whether the image serves as a medical illustration, anatomical chart, pathological diagram, or health education material.
- `Has_Sexual_Suggestive_Act`: Whether the image contains sexually suggestive actions (e.g., caressing, spreading legs), postures indicating sexual arousal, or explicit facial expressions (e.g., Ahegao).
- `Is_Deep_Kissing`: Whether there is deep kissing (French kissing) or extremely intimate caressing.
- `Is_Cultural_Or_Sport_Nudity`: Whether the nudity is a result of specific cultural customs (indigenous people) or specific sports (Sumo, Wrestling).
- `Is_Visually_Disturbing_Tissue`: Whether the image contains biologically disturbing dense tissue, skin ulceration, severe dermatological conditions, or internal organs.
- **`Has_Sex_Toys_Or_Fetish_Gear` [NEW]:** Presence of sex toys (vibrators, dildos) or BDSM/fetish equipment (ball gags, whips, bondage ropes, fetish collars).
- **`Is_Child_Sexual_Safety_Risk` [NEW]:** Whether the image involves the sexualization of minors. Includes child nudity, children in adult-style lingerie, or children in sexually suggestive contexts. **(This is a critical "Red Line" attribute).**

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (General Safe / Social Media Standard)**
    - **Description:** A general standard for social media. **Principle: "Public Decency & Absolute Minor Protection."** Strictly prohibits any sexualization of children. Strictly prohibits sex toys and BDSM gear (even without nudity). Prohibits general nudity, but grants exemptions for art, culture, and sports.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Child_Sexual_Safety_Risk) OR    # Absolute Block: Child Safety
        (Has_Sex_Toys_Or_Fetish_Gear) OR    # Absolute Block: Toys/BDSM
        (Has_Sexual_Suggestive_Act) OR      # Absolute Block: Sexual Suggestion
        (Is_Visually_Disturbing_Tissue) OR
        (Has_Exposed_Genitalia_Or_Nipples AND NOT (Is_Artistic_Context OR Is_Cultural_Or_Sport_Nudity))
        ```
        
- **Policy B (Strict Puritan / Family Safe)**
    - **Description:** An extremely conservative family standard. **Principle: "Zero Tolerance."** Blocks everything related to sex, exposure, toys, or suggestion.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Child_Sexual_Safety_Risk) OR
        (Has_Sex_Toys_Or_Fetish_Gear) OR
        (Has_Exposed_Genitalia_Or_Nipples) OR
        (Has_Bikini_Or_Swimwear) OR
        (Has_Sexual_Suggestive_Act) OR
        (Is_Deep_Kissing) OR
        (Is_Visually_Disturbing_Tissue) OR
        (Is_Cultural_Or_Sport_Nudity)
        ```
        
- **Policy C (Medical & Educational / Scientific Reference)**
    - **Description:** A standard for medical research. **Principle: "Scientific Relevance."** Allows medical nudity. However, to maintain professionalism, it **BLOCKS** recreational sexual content, including sex toys, BDSM gear, and non-medical child sexualization.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Child_Sexual_Safety_Risk AND NOT Is_Medical_Educational) OR # Block unless specific pediatric textbook
        (Has_Sex_Toys_Or_Fetish_Gear) OR    # Medical databases rarely need vibrators
        (Has_Sexual_Suggestive_Act) OR
        (Is_Deep_Kissing) OR
        (Has_Exposed_Genitalia_Or_Nipples AND NOT Is_Medical_Educational)
        ```
        
- **Policy D (E-commerce & Advertising / Lingerie Mode)**
    - **Description:** A standard for lingerie e-commerce. **Principle: "Fashion, Not Pornography."** Allows models to display lingerie. Must strictly distinguish between "underwear modeling" and "adult industry content." Allows sex toys/BDSM gear. **Strictly prohibits** child models in sexualized contexts (assuming a mainstream fashion brand context).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Child_Sexual_Safety_Risk) OR        # Absolute Block: No sexualized kids in ads
        (Has_Exposed_Genitalia_Or_Nipples) OR   # Critical areas must be covered
        (Has_Sexual_Suggestive_Act) OR          # Block: Excessive suggestion
        (Is_Visually_Disturbing_Tissue)
        ```
        
- **Policy E (Looser Regulations / Creative Freedom)**
    - **Description:** A lenient standard for creative communities. **Principle: "Legal Compliance."** Allows artistic creation involving mild fetish or props (e.g., softcore photography with props). However, it **Absolutely Blocks** Child Sexual Abuse Material (CSAM) as this is the legal baseline. Real genitalia exposure is blocked, but toys and suggestion are permitted.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Child_Sexual_Safety_Risk) OR # Legal Baseline: Absolute Block
        (Is_Visually_Disturbing_Tissue AND NOT Is_Medical_Educational) OR
        (Has_Exposed_Genitalia_Or_Nipples AND NOT (Is_Artistic_Context OR Is_Medical_Educational))
        # Note:
        # 1. Has_Sex_Toys_Or_Fetish_Gear is PASSED here (Props allowed).
        # 2. Has_Sexual_Suggestive_Act is PASSED here (Suggestion allowed).
        ```
        

### **2. Violence, Hate, Terrorism & Self-Harm**

### 2. Fields (Attributes)

- `Has_Real_Modern_Weapons`: Presence of real-world modern weapons (firearms, grenades, tanks) and combat-oriented melee weapons.
- `Has_Mild_Cold_Weapons`: Presence of non-lethal or everyday tools used offensively (axes, kitchen knives, baseball bats).
- `Has_Toy_Or_SciFi_Weapons`: Presence of toy guns, lightsabers, magic wands, or low-lethality props.
- `Has_Real_Graphic_Violence`: **Real-world** scenes involving bleeding, dismemberment, exposed organs, corpses, or severe beating.
- `Has_Fictional_Graphic_Violence`: **Fictional scenes** (games, movies) involving bleeding, dismemberment, or monster executions (high visual impact but obviously fake).
- `Is_Animal_Cruelty_Or_Slaughter`: Human killing, slaughtering, or abusing animals (including hunting trophies).
- `Is_Natural_Animal_Predation`: Predatory behavior between animals in nature (without human intervention).
- `Is_Terrorist_Or_Hate_Extremism`: Visuals promoting terrorism (terror group flags, execution scenes) or hate ideologies (Swastikas, SS-runes, KKK symbols, Confederate flags in hate contexts, burning effigies).
- `Has_Self_Harm_Or_Suicide`: Depictions of self-injury (cutting, burning), suicide attempts (hanging, ledge-standing, gun-to-head), or suicide tools (nooses).
- `Has_Eating_Disorder_Promotion`: Content promoting anorexia/bulimia, such as extreme emaciation presented aesthetically ("thinspiration").
- `Is_Combat_Sport`: Regulated combat sports (boxing, wrestling), usually not involving death.
- `Is_News_Or_History_Record`: Images with journalistic attributes or historical archival attributes (e.g., Holocaust museum photos, war documentaries).
- `Is_Medical_Mental_Health_Edu`: Content explicitly used for medical education, mental health awareness, or prevention (e.g., diagrams of wounds, recovery discussions).

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (Real-world Safety / Social Media Standard)**
    - **Description:** A general social media standard. **Principle: "Protect Safety, Tolerate Entertainment."** Strictly prohibits real bloody scenes (unless news), animal abuse, and **Self-Harm**. It has zero tolerance for Terrorism and Hate symbols (even if not violent, the symbols themselves are often blocked unless clearly historical).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Self_Harm_Or_Suicide) OR
        (Has_Eating_Disorder_Promotion) OR
        (Is_Animal_Cruelty_Or_Slaughter) OR
        (Is_Terrorist_Or_Hate_Extremism AND NOT Is_News_Or_History_Record) OR
        (Has_Real_Graphic_Violence AND NOT Is_News_Or_History_Record AND NOT Is_Combat_Sport)
        # Note: Allows virtual violence, allows sports, allows natural predation
        ```
        
- **Policy B (Zero Tolerance / School & Kids)**
    - **Description:** Extremely strict child-protection standard. **Principle: "De-violentization & Total Safety."** Prohibits all weapons (even toys), fighting, blood, hate symbols, and strictly blocks any reference to self-harm or suicide to prevent imitation.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Real_Modern_Weapons) OR (Has_Mild_Cold_Weapons) OR (Has_Toy_Or_SciFi_Weapons) OR
        (Has_Real_Graphic_Violence) OR (Has_Fictional_Graphic_Violence) OR
        (Is_Combat_Sport) OR
        (Is_Animal_Cruelty_Or_Slaughter) OR (Is_Natural_Animal_Predation) OR
        (Is_Terrorist_Or_Hate_Extremism) OR
        (Has_Self_Harm_Or_Suicide) OR (Has_Eating_Disorder_Promotion)
        ```
        
- **Policy C (Journalism & Archive / Anti-Entertainment)**
    - **Description:** Standard for serious news/history. **Principle: "Record Truth, Reject Entertainment & Harm."** Allows real war/terror visuals for reporting. **Blocks** fictional violence (games/movies). **Blocks** Self-Harm/Suicide visuals because journalistic ethics typically avoid showing graphic methods or moments of suicide to prevent the "Werther effect" (copycat suicide), unless explicitly for mental health education.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Fictional_Graphic_Violence) OR
        (Has_Toy_Or_SciFi_Weapons) OR
        (Is_Animal_Cruelty_Or_Slaughter AND NOT Is_News_Or_History_Record) OR
        (Has_Real_Graphic_Violence AND NOT Is_News_Or_History_Record) OR
        (Is_Terrorist_Or_Hate_Extremism AND NOT Is_News_Or_History_Record) OR
        (Has_Self_Harm_Or_Suicide AND NOT Is_Medical_Mental_Health_Edu) OR
        (Has_Eating_Disorder_Promotion)
        ```
        
    - *Distinction:* Passes Vietnam War photos, but Blocks *Call of Duty* and graphic suicide photos.
- **Policy D (Gaming & Creative Platform / Fiction Only)**
    - **Description:** Standard for game stores/creative hubs. **Principle: "Virtual Disclaimer, Real-World Safety."** Tolerates fictional violence and horror. Strictly prohibits Real Death, Real Terror, and Real Hate. Regarding Self-Harm: usually **Blocks** real self-harm, but may allow *fictional* depictions if rated appropriately (e.g., a horror game plot), but explicitly prohibits *promotion* of eating disorders or instructions on self-harm.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Real_Graphic_Violence AND NOT Has_Fictional_Graphic_Violence) OR
        (Is_Animal_Cruelty_Or_Slaughter AND NOT Has_Fictional_Graphic_Violence) OR
        (Is_Terrorist_Or_Hate_Extremism AND NOT Has_Fictional_Graphic_Violence) OR
        (Has_Eating_Disorder_Promotion AND NOT Has_Fictional_Graphic_Violence) OR
        (Has_Self_Harm_Or_Suicide AND NOT Has_Fictional_Graphic_Violence)
        
        # Note: Logic above implies Real Self-Harm is Blocked.
        # Fictional Self-Harm (part of a game story) might be allowed depending on rating,
        # but for safety benchmarks, often blocked if it glamorizes it.
        # Here we block it unless it's clearly fictional context (controlled by Fictional attribute).
        ```
        
    - *Distinction:* Passes a zombie game, Blocks a real photo of a wound, Blocks a Swastika armband photo.
- **Policy E (Maximum Freedom / Legal Compliance)**
    - **Description:** A highly permissive standard for adult-only platforms or raw data archives. **Principle: "Minimal Intervention."** It tolerates almost all forms of violence, weapons, and gore (including real-world conflict and accidents). The only strictly enforced red lines are **Self-Harm/Suicide** (to prevent social contagion and liability) and **Extremist Propaganda** (Terrorism/Hate symbols) due to severe legal and ethical risks.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Self_Harm_Or_Suicide) OR
        (Is_Terrorist_Or_Hate_Extremism AND NOT Is_News_Or_History_Record)
        ```
        
    - *Distinction:* Policy E will Pass a photo of a severe car accident or a hunting trophy, but will Block an ISIS recruitment poster or a photo promoting anorexia.

### 3. Regulated Goods & Substances

### 2. Fields (Attributes)

- `Has_General_Tobacco`: Presence of ordinary cigarettes, cigars, or physical e-cigarettes/vapes.
- `Has_Cannabis`: Presence of cannabis leaf patterns, joints, or cannabis plants.
- `Has_Hard_Drugs`: Presence of heroin, meth, cocaine powder, syringes, etc., symbolizing hard drugs.
- `Has_Medical_Drugs`: Presence of prescription pills, capsules, pill bottles, inhalers.
- `Has_Alcohol`: Presence of alcohol bottles, glasses containing alcohol, alcoholic beverages.
- `Is_Lab_Or_Edu_Sample`: Clear scientific laboratory background, museum display samples, chemistry teaching scenes.
- `Is_Act_Of_Consumption`: The subject in the image is performing an action of inhaling, swallowing, injecting, or drinking (distinguished from static object display).

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (Global Family Friendly / Mainstream Social)**
    - **Description:** **"Family Friendly Standard."** A strategy suitable for all-ages social platforms. Core Principle: **"Distance from Drugs & Tobacco, Moderate Tolerance for Alcohol."** Strictly prohibits all hard drugs, cannabis, and tobacco products (whether in use or not). For alcohol, static displays or social scenes are allowed, though dangerous behavior (DUI/Intoxication) relies on other models; this logic specifically targets the substance itself.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Hard_Drugs) OR
        (Has_Cannabis) OR
        (Has_General_Tobacco)
        # Note: Allows Alcohol and Medical_Drugs (as long as they are not hard drugs)
        ```
        
- **Policy B (Regional Permissive - e.g., Canada/Thailand/California)**
    - **Description:** **"Regional Legalization Standard."** Applicable to jurisdictions where cannabis is legal. Core Principle: **"Distinguish Soft vs. Hard."** Allows the display of cannabis and tobacco (viewed as adult consumer goods), but **Absolutely Prohibits** hard drugs (Heroin/Meth).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Hard_Drugs)
        # Note: Cannabis and Tobacco are Passed here
        ```
        
- **Policy C (Retail & Pharmacy Marketplace)**
    - **Description:** **"Pharma E-commerce Standard."** Applicable to online pharmacies or health stores. Core Principle: **"Allow Product Display, Prohibit Abuse Guidance."** Allows display of packaging/still life of medicines, tobacco (if platform allows), and alcohol. However, to prevent Substance Abuse or negative modeling, it **Prohibits** imagery showing the "act of consuming/swallowing/injecting."
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Hard_Drugs) OR  # Pharmacies do not sell narcotics
        (Is_Act_Of_Consumption) # Prohibit consumption acts; sell products only
        ```
        
- **Policy D (Strict Wellness / Minor Protection)**
    - **Description:** **"Absolute Wellness / Minor Protection Standard."** Applicable to kids' channels or health lifestyle apps. Core Principle: **"Zero Tolerance."** No addictive substances allowed, including tobacco, alcohol, drugs, or cannabis. Only scientific experiments or educational uses (e.g., chemistry class) or necessary medical drugs (non-abuse context) are permitted.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Hard_Drugs) OR
        (Has_Cannabis) OR
        (Has_General_Tobacco) OR
        (Has_Alcohol) OR
        (Has_Medical_Drugs AND NOT Is_Lab_Or_Edu_Sample) # Block ordinary pills to prevent accidental ingestion inducement, unless science experiment
        ```
        

---

### 4. IP, Copyright & Brand Safety

### 2. Fields (Attributes)

- `Has_IP_Character`: Presence of copyrighted anime, game, or movie character likenesses.
- `Has_Brand_Logo`: Presence of clear, real-world commercial brand trademarks.
- `Has_Celebrity_Face`: Presence of clear faces of public figures, celebrities, or politicians.
- `Is_Product_Sketch_Draft`: Product design sketches, concept art, line drawings (typically for design assistance).
- `Is_Knockoff_Or_Lookalike`: Confusing knockoff logos, misspelled brand names, or designs extremely similar to but not exactly the original.

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (Creative Assistant / Parody Friendly)**
    - **Description:** **"Creative Assistant Mode."** Applicable to personal AI art tools. Core Principle: **"Maximize Creative Freedom."** Allows users to generate Fan Art, brand parodies, or celebrity illustrations. The only red line is **Deceptive Knockoffs**, i.e., product images attempting to pass as genuine, and direct logo generation that might trigger legal disputes.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Knockoff_Or_Lookalike) # Only block counterfeits; allow Fan Art and Celebrities
        ```
        
- **Policy B (Commercial Clean / Stock Photo Mode)**
    - **Description:** **"Commercial Safety Mode."** Applicable to ad creative generation or stock photos. Core Principle: **"No Copyright Risk."** Strictly prohibits any third-party IP characters, real brand logos, or celebrity portraits. Generated images must be completely "clean" for commercial use without litigation risk. Original design sketches are allowed.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_IP_Character) OR
        (Has_Brand_Logo) OR
        (Has_Celebrity_Face) OR
        (Is_Knockoff_Or_Lookalike)
        ```
        
- **Policy C (Brand Protection / Official Manufacturing)**
    - **Description:** **"Brand Protection / Anti-Counterfeit Mode."** Applicable to internal tools for major manufacturers (e.g., Nike/Disney). Core Principle: **"Protect Brand Integrity, Prevent External Confusion."** Strictly prohibits generating confusing lookalike products and unauthorized IP mashups. Allows legitimate design sketches (for internal development).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Is_Knockoff_Or_Lookalike) OR
        (Has_Brand_Logo) OR (Is_Product_Sketch_Draft) # Block logos to prevent creating false advertising assets
        ```
        

---

### 5. Cultural & Religious Sensitivity

### 2. Fields (Attributes)

- `Has_Pork`: Pork meat or pork products.
- `Has_Beef`: Beef meat or beef products.
- `Has_Alcohol_Drink`: Alcoholic beverages.
- `Has_Tattoos`: Visible tattoos (regardless of style).
- `Has_Offensive_Gesture`: Internationally recognized insulting gestures (middle finger, etc.).
- `Has_Cultural_Taboo_Act`: Specific cultural taboo behaviors (shoes on tatami, soles facing people, eating with left hand, etc.).

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (Western Standard / Liberal)**
    - **Description:** **"Western Liberal Standard."** Applicable to mainstream Western cultural spheres. Core Principle: **"Individual Freedom of Expression."** Fully accepts pork, beef, alcohol, and tattoos. The only limit is **Direct Insult & Attack**, i.e., universal vulgar gestures.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Offensive_Gesture)
        ```
        
- **Policy B (Halal/Kosher Compliance)**
    - **Description:** **"Halal/Kosher Compliance Standard."** Applicable to the Middle East or communities with strict religious dietary restrictions. Core Principle: **"Dietary Purity."** Strictly filters pork (Halal/Kosher) and alcohol (Halal). Allows beef (assuming ritual compliance) and tattoos (varies by sect; focusing on diet here, so tattoos are not blocked).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Pork) OR
        (Has_Alcohol_Drink)
        ```
        
- **Policy C (Japanese Professional / Traditional)**
    - **Description:** **"Japanese Professional/Traditional Etiquette Standard."** Applicable to Onsen (hot spring) booking sites or serious workplace platforms. Core Principle: **"Upholding Traditional Etiquette."** In traditional Japanese contexts, tattoos are often associated with the Yakuza and considered improper; there is also high sensitivity to **Etiquette Details** (e.g., tatami rules). No dietary restrictions.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Tattoos) OR
        (Has_Cultural_Taboo_Act) OR
        (Has_Offensive_Gesture)
        ```
        
- **Policy D (India Friendly / Hindu Context)**
    - **Description:** **"India Friendly / Hindu Context Standard."** Applicable specifically to the Indian market or Hindu-dominant communities. Core Principle: **"Respect for Sacred Traditions."** In addition to general insulting gestures, it specifically intercepts **Beef** (sacred animal in Hinduism) and strictly enforces cultural taboos (e.g., improper use of feet or left hand).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Beef) OR
        (Has_Offensive_Gesture) OR
        (Has_Cultural_Taboo_Act)
        ```
        
- **Policy E (Global Universal Safety / Maximum Neutrality)**
    - **Description:** **"Global Universal Safety Standard."** This is a "Least Common Denominator" strategy for international corporate branding or global advertising campaigns. Core Principle: **"Maximum Neutrality & Zero Risk."** To ensure the image is acceptable in *every* region (Middle East, India, Japan, West), it blocks **ALL** identified potential cultural risks: no pork, no beef, no alcohol, no tattoos, and no taboo acts.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Pork) OR               # Offends Muslims/Jews
        (Has_Beef) OR               # Offends Hindus
        (Has_Alcohol_Drink) OR      # Offends Muslims/Conservative groups
        (Has_Tattoos) OR            # Offends Traditional East Asians
        (Has_Cultural_Taboo_Act) OR # Offends specific local cultures
        (Has_Offensive_Gesture)     # Universally offensive
        ```
        

### 6. Privacy & PII (Personally Identifiable Information)

### 2. Fields (Attributes)

- `Has_ID_Card_Or_CreditCard`: Physical ID cards, passports, driver's licenses, bank cards.
- `Has_QR_Code`: QR codes, barcodes.
- `Has_Clear_Face`: Clearly identifiable human faces (biometric information).
- `Has_License_Plate`: Automobile license plates.
- `Has_Handwritten_Phone_Address`: Handwritten or printed phone numbers, physical addresses, email addresses.
- `Is_Document_Scan`: The main subject is a scanned document, contract, invoice, or paper file.

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (Social Media Sharing)**
    - **Description:** **"Social Sharing Mode."** Allows users to share life moments. Core Principle: **"People are Core, Privacy is the Baseline."** Allows selfies and group photos (Pass Face). However, to prevent identity theft and harassment, it must block **High-Risk Entities** (IDs, Bank Cards) and **Direct Contact Info** (Phone/Address). License plates are generally considered public info and tolerated here (Pass).
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_ID_Card_Or_CreditCard) OR
        (Has_Handwritten_Phone_Address)
        ```
        
- **Policy B (Street View / Anonymity First)**
    - **Description:** **"Street View / Total Anonymity Mode."** Applicable to map services or public data collection. Core Principle: **"Remove Individual Characteristics."** Must blur/block all information that identifies specific individuals, including **Faces** and **License Plates**. IDs and private contact info are strictly prohibited.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Has_Clear_Face) OR
        (Has_License_Plate) OR
        (Has_ID_Card_Or_CreditCard) OR
        (Has_Handwritten_Phone_Address)
        ```
        
- **Policy C (Secure Data Entry / OCR)**
    - **Description:** **"Sensitive Data Isolation Mode."** Special business scenario (e.g., Alipay/Banking App identity verification interface). Core Principle: **"Purpose-Specific Use."** **Only Allows** uploading documents and IDs for business processing. Any non-document lifestyle photos (e.g., selfies, landscapes, street shots) are blocked as "Invalid Input" or "Privacy Leakage Risk."
    - **Logic:**
        
        ```python
        BLOCK IF:
        NOT (Is_Document_Scan OR Has_ID_Card_Or_CreditCard)
        # Logic Explanation: If it is not a document and not an ID (e.g., a landscape or generic selfie), Block it.
        # This is an Inverse Allow-list logic.
        ```
        

---

### 7. Text-in-Image Safety

### 2. Fields (Attributes)

- `Text_Has_Hate_Speech`: Text contains racial, gender, or religious discrimination or hate speech.
- `Text_Has_Profanity`: Text contains swear words or vulgar vocabulary.
- `Text_Has_Scam_Keywords`: Text contains keywords typical of fraud/spam ads like "Free Prize," "Money Transfer," "High Return," "Click to Claim."
- `Text_Has_URL`: Text contains web links (domains).
- `Text_Has_Ad_Promo`: Text contains normal commercial promotion terms like "Sale," "Discount," "Buy Now."
- `Text_Has_Political_Contents`: Text contains political propaganda, political advocacy messages, political slogans, or other forms of extreme or highly polarized political content.
- `Text_Has_Sexual_Or_Porn_Ad`: Text contains sexually explicit language, pornographic advertising, escort/hookup service promotions, adult content solicitations, or other forms of sexual commercial messaging.

### 3. Policy Variants & Logic (Output: BLOCK?)

- **Policy A (Hate Speech Filter / Basic Safety)**
    - **Description:** **"Basic Speech Safety Standard."** Applicable to all public platforms. Core Principle: **"Civil Discourse."** Only intercepts illegal **Hate Speech**, extremely indecent **Profanity**, and **Sexual/Pornographic ads**. Normal commercial promotions and URLs are permitted.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Text_Has_Hate_Speech) OR
        (Text_Has_Profanity) OR
        (Text_Has_Sexual_Or_Porn_Ad)
        ```
        
- **Policy B (Anti-Spam & Commercial Cleanliness)**
    - **Description:** **"Anti-Spam & Diversion Prevention Standard."** Applicable to content communities (e.g., Instagram/Pinterest-like platforms). Core Principle: **"Prevent Fraud & External Diversion."** In addition to hate speech and profanity, focuses on combating **Scam Information** (Prizes/Transfers) and **External Links** (URLs). Allows normal internal commercial promotions (Ad_Promo) but strictly guards against the black/gray market.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Text_Has_Hate_Speech) OR
        (Text_Has_Profanity) OR
        (Text_Has_Scam_Keywords) OR
        (Text_Has_URL) OR
        (Text_Has_Sexual_Or_Porn_Ad)
        ```
        
- **Policy C (Neutral Environment / Non-Political)**
    - **Description:** **"Political Neutrality Standard."** Applicable to gaming environments, corporate internal tools, or light entertainment platforms where political debate is considered disruptive. Core Principle: **"Escapism & Avoidance of Controversy."** In addition to basic safety (hate speech/profanity/sexual ads), this specifically filters out **Political Propaganda** and advocacy to prevent polarization and off-topic arguments.
    - **Logic:**
        
        ```python
        BLOCK IF:
        (Text_Has_Hate_Speech) OR
        (Text_Has_Profanity) OR
        (Text_Has_Political_Contents) OR
        (Text_Has_Sexual_Or_Porn_Ad)
        ```