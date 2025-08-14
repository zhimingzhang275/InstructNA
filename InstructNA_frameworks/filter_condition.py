import nupack
def is_valid_three_way_junction(structure: str) -> bool:
    """
    检查三通结构中是否有 >2 个分支且由 >1 nt bulge 分隔。
    """
    import re
    
    # 把所有的 stem 区段 (....) 找出来
    stems = list(re.finditer(r'\(+\.{1,}\)+', structure))
    
    if len(stems) < 3:
        return True  # 不足3个stem，不构成三通结构

    # 提取所有stem的位置
    stem_positions = [(m.start(), m.end()) for m in stems]

    # 计算任意两个相邻stem之间的bulge长度
    valid_bulges = 0
    for i in range(len(stem_positions) - 1):
        end1 = stem_positions[i][1]
        start2 = stem_positions[i + 1][0]
        bulge_len = start2 - end1
        if bulge_len > 1:
            valid_bulges += 1
    
    # 需要至少2个bulge长度 >1 nt 的隔开
    return valid_bulges >= 2


# def check_conditions_1(seq):
#     "宽松支架条件检查：P1和P3的5'端必须是GCCG，且P1和P3的3'端必须是互补的。"
#     model = nupack.Model(material="dna", celsius=25,sodium=0.14,magnesium=0.005)  
#     result = nupack.mfe(strands=[seq], model=model)  
#     ss=str(result[0].structure)
#     P1_c1=ss[1:6]
#     P1_c2=ss[-5:]
#     P3_c1=ss[17:20]
#     P3_c2=ss[-8:-5]
#     if (set(P1_c1).issubset({'('}) or set(P1_c2).issubset({')'})) \
#     and (set(P3_c1).issubset({'('}) or set(P3_c2).issubset({')'})) and seq[16:20]=="GCCG":
#         return ss
#     return False



# def check_conditions_T4(seq):
    
#     "宽松支架条件检查：P1和P3的5'端必须是GCCG，且P1和P3的3'端必须是互补的。"
#     model = nupack.Model(material="dna", celsius=25,sodium=0.14,magnesium=0.005)  
#     result = nupack.mfe(strands=[seq], model=model)  
#     ss=str(result[0].structure)
#     P1_c1=ss[1:6]
#     P1_c2=ss[-5:]
#     P3_c1=ss[17:20]
#     P3_c2=ss[-12:-9]
#     if (set(P1_c1).issubset({'('}) and set(P1_c2).issubset({')'})) \
#     and (set(P3_c1).issubset({'('}) and set(P3_c2).issubset({')'})) \
#     and seq[16:20]=="GCCG" \
#     and seq[28:31]=="CGG" \
#     and seq[7] !="G" \
#     and is_valid_three_way_junction(ss):
#         return ss
#     print(f"Sequence {seq} does not meet the conditions.")
#     return False

def check_conditions(seq):
    
    "宽松支架条件检查：P1和P3的5'端必须是GCCG，且P1和P3的3'端必须是互补的。"
    model = nupack.Model(material="dna", celsius=25,sodium=0.14,magnesium=0.005)  
    result = nupack.mfe(strands=[seq], model=model)  
    ss=str(result[0].structure)
    P1_c1=ss[1:6]
    P1_c2=ss[-5:]
    P3_c1=ss[16:19]
    P3_c2=ss[30:33]
    if (set(P1_c1).issubset({'('}) and set(P1_c2).issubset({')'})) \
    and (set(P3_c1).issubset({'('}) and set(P3_c2).issubset({')'})) \
    and seq[16:20]=="GCCG" \
    and seq[30:33]=="GGC" \
    and is_valid_three_way_junction(ss):
        return ss
    print(f"Sequence {seq} does not meet the conditions.")
    return False