# Original Dataset (400)
- all ages = 100
- children = 100
- adolescent = 100
- adult = 100

# Synthesized Dataset (4006)
## Input original dataset -> Generate Indonesian Songs
- all ages = 1000
- children = 1002
- adolescent = 1003
- adult = 1001

# Generated Dataset (4003)
## Generate English Songs (more creative lyrics) -> Translate to Indonesian
- all ages = 1003
- children = 1000
- adolescent = 1000
- adult = 1000

# Generated Dataset 2 (4009)
## Generate Indonesian Songs (Faster, but the lyrics are not creative)
- all ages = 1002
- children = 1003
- adolescent = 1001
- adult = 1003

# PREPROCESSED
- original: train + test
- synthesized: train + test
- generated: train + test
- full_combination (original + synthesized + generated): train + test
- split_combination: train (synthesized or generated) + test (original)


![image](https://github.com/user-attachments/assets/83b21a84-9729-49a8-84a3-d28f9a12ccec)
