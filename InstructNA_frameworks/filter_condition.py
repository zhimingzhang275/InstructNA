import nupack
import re
def is_valid_three_way_junction(structure: str) -> bool:

    stems = list(re.finditer(r'\(+\.{1,}\)+', structure))
    
    if len(stems) < 3:
        return True  

    stem_positions = [(m.start(), m.end()) for m in stems]

    valid_bulges = 0
    for i in range(len(stem_positions) - 1):
        end1 = stem_positions[i][1]
        start2 = stem_positions[i + 1][0]
        bulge_len = start2 - end1
        if bulge_len > 1:
            valid_bulges += 1
    
    return valid_bulges >= 2


def check_conditions(seq):
    
    model = nupack.Model(material="dna", celsius=25,sodium=0.14,magnesium=0.005)  
    result = nupack.mfe(strands=[seq], model=model)  
    ss=str(result[0].structure)
    P1_c1=ss[1:6]
    P1_c2=ss[-5:]
    P3_c1=ss[17:20]
    P3_c2=ss[-8:-5]
    if (set(P1_c1).issubset({'('}) or set(P1_c2).issubset({')'})) \
        and (set(P3_c1).issubset({'('}) or set(P3_c2).issubset({')'})) \
        and seq[16:20]=="GCCG" \
        and is_valid_three_way_junction(ss):
        return ss
    return False
