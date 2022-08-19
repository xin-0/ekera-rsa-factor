from logging import exception
from math import gcd
from typing import Union, Tuple
from pyqpanda import *
import numpy as np
from typing import List


def init_circ(
    q_reg: QVec,
    init_state: str
) -> QCircuit:
    """ 根据 `init_state` 提供的二进制字符串来初始化 `q_reg`
        返回初始化线路。

        init_state 的 size 可以小于等于 q_reg. 
    """

    circ = QCircuit()

    # 处理输入的二进制数字无法用当前量子寄存器表示的情况
    if len(q_reg) < len(init_state):
        raise Exception(f"Not Enought Qubit for init state: {init_state}", len(q_reg))

    l_endian = init_state[::-1]

    # 遍历初始化字符串，在值为 ‘1’ 的位置对应的量子比特上添加 X 门
    for i,c in enumerate(l_endian):
        if c == '1':
            circ << X(q_reg[i])

    return circ



def ry_QFT(
    q_reg: QVec,
    inv: bool = False,
    approx_level: int = 0
) -> QCircuit:
    """ 在 `circ` 的 `q_reg` 上添加由 Ry 门构成的 Quantum Fourier Transformation
        线路。

        Args:            
            q_reg: `List[Qubit]`
                要被添加 ryQFT 的具体量子比特组或量子寄存器

            inv: `bool`
                构建的是否为 inverse QFT。True 的时候为 QFT^(-1)。

            approx_level: `int`
                近似等级，如果为 `i` 则会在构建线路的时候舍弃小于 
                ${\pi \over 2^i}$ 角度的旋转。为 `0` 时表示不做近似。
    """

    yQFT_circ = QCircuit()

    num_q = len(q_reg)

    for ctl_q in reversed(range(num_q - 1)):
        for target in reversed(range(ctl_q + 1, num_q)):
            angle_exp = target - ctl_q
            if approx_level > 0 and angle_exp > approx_level:
                break
            theta = np.pi/(2**angle_exp)
            yQFT_circ << RY(q_reg[target], theta).control([q_reg[ctl_q]])

    if inv:
        return yQFT_circ.dagger()

    return yQFT_circ



def ctl_phi_adder(
    X: Union[int, float],
    ctl_q: Qubit,
    target: QVec,
    truncate: bool = False
) -> QCircuit:
    """
        在进行了RY-QFT 变换的空间里做加法运算。结果仍在 RY-QFT 空间中。
        加法的执行受控于控制比特 `ctl_q`。

        当 ctl_q = 0, `target` 维持原来的数值。

        当 ctl_q = 1, `target` --> `target + X`。

        Args:
            X: `int | float`
                加数，可以为任意整数。为负时即等同于做减法。经典数据。

            ctl_q: `Qubit`
                控制比特

            target: `QVec`
                被加数，在 RY-QFT内表示。量子数据

            truncate: `bool`
                是否去除过于小的旋转角度。
    """

    phi_add_circ = QCircuit()
    
    if X == 0:
        return QCircuit()

    precision = len(target)
    if truncate:
        precision = min(int(np.log2(precision*abs(X)))+1, precision)
    

    for k in range(precision):
        theta = np.pi*X/(2**(k))
        phi_add_circ << RY(target[k], theta).control([ctl_q])

    return phi_add_circ



def phi_MAC(
    ctl_reg: QVec,
    targ_reg: QVec,
    X: int,
    N: int
) -> Tuple[QCircuit, List[int]]:
    """
        phi-MAC(X|N): 
            quantum multiply-accumulate circuit (MAC) in Fourier Base.
            
            乘法累加器，用于计算 X*z mod N 的初始近似值 t:
            `t = \sum_{k=0}^{n-1}{z_k *(2^{k} * X \mod{N})}`
            上式计算出的 t 满足 

                1. `0 <= t < n*N`
                2. `t = X*z mod N`
            
            该量子线路实现以下计算过程：

            |z>|y>^(Phi) --- Phi_MAC(X|N) --> |z>|t+y>^(Phi)

                |z> 储存在 `ctl_reg`里。|y>^(Phi) 储存在 `targ_reg`。

                ^(Phi)表示寄存器代表的数值处于傅立叶basis。

        Args:
            ctl_reg: `QVec`
                存储被乘数 `z` 的量子寄存器，量子数据。

            targ_reg: `QVec`
                存储要累加的对象 `y` 的量子寄存器，量子数据。

            X: `int`
                乘数，经典数据。
                
            N: `int`
                模数，经典数据。
        
        Returns:
            QCircuit:
                phi-MAC(X|N)的量子线路
            
            List[int]:
                线路中 z_0,... z_n 每个单比特所控制的 `ctl_phi_adder` 所使用的加数。
                方便之后调用，减少经典计算量。
    """

    phi_mac_circ = QCircuit()

    addant_list = [X%N]
    for i in range(1, len(ctl_reg)):
        new_addant = (addant_list[i-1] * 2) % N
        addant_list.append(new_addant)
    
    for i, addant in enumerate(addant_list):
        phi_mac_circ << ctl_phi_adder(addant, ctl_reg[i], targ_reg)


    return phi_mac_circ, addant_list



def phi_MAC_list(
    ctl_reg: QVec,
    targ_reg: QVec,
    X_list: List[int]
) -> QCircuit:
    """
        与 `phi-MAC` 完成相同的计算，区别是提前计算好线路中 
        z_0,... z_n 每个单比特所控制的 `ctl_phi_adder` 所使用的加数，
        并将其作为 `X_list` 传递进来。列表`X_list`中的为经典数据。
    """
    
    phi_mac_list_circ = QCircuit()

    for i, X in enumerate(X_list):
        phi_mac_list_circ << ctl_phi_adder(X, ctl_reg[i], targ_reg)

    return phi_mac_list_circ



def perm_last2top(
    q_reg: QVec
) -> QCircuit:
    """
        将量子寄存器 `q_reg` 中处于最高位的量子比特置换到最低位，
        其余量子比特之间的相对位置不变，即:

        |q0>      ----->   |q(n-1)>
        |q1>      ----->   |q0> 
         ·        ----->   |q1>
         ·          ·       ·
         ·          ·       ·
        |q(n-2)>  ----->    ·
        |q(n-1)>  ----->   |q(n-2)>
    """

    permute_circ = QCircuit()

    for i in reversed(range(1,len(q_reg))):
        permute_circ << SWAP(q_reg[i], q_reg[i-1])
    
    return permute_circ


def OOP_MM(
    z: QVec,
    zero_mp1: QVec,
    zero_n: QVec,
    M: int,
    N: int,
    reverse: bool = False,
    inverse_M: bool = False,
    approx_level: int = 0
) -> QCircuit:
    """ 
        Out-of-place (OOP) quantum-classical modular multiplication (MM)

        Out-of-place 的模乘法量子线路。要求 `M` 与 `N` 互质。
        实现以下计算过程:

        |z>|0> --- OOP_MM(M,N) ---> |z>|M*z mod N>

        名称为 "out-of-place" 是因为作为结果的 `M*z mod N` 没有覆写到
        `z` 的量子寄存器上。

        Args:
            z: `QVec`
                模乘法的被乘数所在的量子寄存器，量子数据。

            zero_mp1: `QVec`
                第一组0寄存器，用于辅助计算。
            
            zero_n: `QVec`
                第二组0寄存器，用于辅助计算。
            
            M: `int`
                模乘法的乘数，经典数据。

            N: `int`
                模乘法的模数，经典数据。

            reverse: `bool`
                为 `True` 的时候会输出 `OOP_MM.dagger`, `OOP_MM` 的反向线路。

            inverse_M: `bool`
                为 `True` 时，乘数变为原来乘数的模乘运算的逆元，即 `M^(-1) mod N`。
                为保证该逆元存在，输入的 `M` 与 `N` 必须互质

            approx_level: `int`
                近似等级，传递给函数内部的傅立叶变换使用。
    """

    n = len(z)
    m = len(zero_mp1) - 1
    
    if len(zero_n) != n:
        raise Exception("Ancilla zero register zero_n's size not correct !", len(zero_n))
    
    if m != int(np.ceil(np.log2(n))):
        raise Exception("Ancilla zero register zero_mp1's size not correct !", len(zero_mp1))

    if gcd(M,N) != 1:
        raise Exception("Input (M, N) pair not co-prime, invalid !", M)

    ##### 确定乘数

    if inverse_M:
        # 求在模为 N 时 M 的乘法逆元
        # 在经典算法里可以高效求出，无overhead。
        M = pow(M, -1, N)

    circ_OOPMM = QCircuit()

    two_mModN = pow(2, m, N)
    M_prime = (M * two_mModN) % N

    ##### 计算 t'

    circ_acc_tPrime, zM_prime_accs = phi_MAC(
        ctl_reg   = z,
        targ_reg  = zero_mp1[:] + zero_n[:],
        X         = M_prime,
        N         = N
    )
    
    ##### 近似计算 (M * z mod N)

    circ_est = QCircuit()
    for i in range(m):
        circ_est << ctl_phi_adder(
            X      = -N/2,
            ctl_q  = zero_mp1[i],
            target = zero_mp1[i+1:] + zero_n[:]
        )
    
    ##### 修正近似

    # 确认是否多减了 N
    circ_extract_sign_qft_dagger = ry_QFT(
        q_reg        = [zero_mp1[-1]] + zero_n[:],
        inv          = True,
        approx_level = approx_level
    )
    
    # 返回 Fourier basis 进行计算
    circ_back2qft = ry_QFT(
        q_reg        = [zero_mp1[-1]] + zero_n[:-1:],
        inv          = False,
        approx_level = approx_level
    )
    
    circ_perm2top = perm_last2top(
        [zero_mp1[-1]] + zero_n[:]
    )

    # 进行条件性修正
    circ_ctl_correction = ctl_phi_adder(
        X      = N,
        ctl_q  = zero_mp1[-1],
        target = zero_n
    )

    circ_ucorrect = QCircuit()
    circ_ucorrect << CNOT(zero_n[0], zero_mp1[-1])

    ##### Uncompute
    #  变换 u_(m+1) 到傅立叶空间为 uncompute 做准备
    circ_u_mp1_qft = ry_QFT(
        q_reg        = zero_mp1,
        inv          = False,
        approx_level = approx_level
    )

    # 将结果变换出傅立叶空间得到最终结果
    circ_Mz_modN_qft_dagger = ry_QFT(
        q_reg        = zero_n,
        inv          = True,
        approx_level = approx_level
    )

    # uncompute u_(m+1)^(Phi)
    R_mp1 = 2**(m+1)
    Ninv_mod_Rmp1 = pow(N, -1, R_mp1)
    for i in range(len(zM_prime_accs)):
        zM_prime_accs[i] = (-1) * ((zM_prime_accs[i] * Ninv_mod_Rmp1) % R_mp1)

    circ_ump1_uncompute = phi_MAC_list(
        ctl_reg  = z,
        targ_reg = zero_mp1,
        X_list   = zM_prime_accs
    )

    # 填充量子线路
    circ_OOPMM  << circ_acc_tPrime << circ_est << circ_extract_sign_qft_dagger \
                << circ_back2qft << circ_perm2top << circ_ctl_correction \
                << circ_ucorrect << circ_u_mp1_qft << circ_Mz_modN_qft_dagger \
                << circ_ump1_uncompute

    if reverse:
        return circ_OOPMM.dagger()

    return circ_OOPMM

def ctl_reg_swap(
    ctl_q: Qubit,
    targ_reg1: QVec,
    targ_reg2: QVec,
    zero_ctl: bool = False
) -> QCircuit:
    """
        根据控制比特的数值，交换两个相同大小的量子寄存器的位置。

        Args:
            ctl_q: `Qubit`
                控制比特

            targ_reg1: `QVec`
                待交换量子寄存器1

            targ_reg2: `QVec`
                待交换量子寄存器2

            zero_ctl: `bool`
                为 `False` 时，只在控制比特为 1 时会交换两量子寄存器。
                为 `True` 时，只在控制比特为 0 时会交换两量子寄存器。
    """
    n = len(targ_reg1)

    if len(targ_reg2) != n:
        raise Exception(f"Register sizes not match!, {len(targ_reg1)} != {len(targ_reg2)}")

    ctl_swap_circ = QCircuit()

    if zero_ctl:
        ctl_swap_circ << X(ctl_q)
    
    for i in range(n):
        ctl_swap_circ << SWAP(targ_reg1[i], targ_reg2[i]).control([ctl_q])
    
    if zero_ctl:
        ctl_swap_circ << X(ctl_q)
    
    return ctl_swap_circ



def ctl_pswap(
    ctl_q: Qubit,
    targ_qubit1: Qubit,
    targ_qubit2: Qubit,
    zero_ctl: bool = False
) -> QCircuit:
    """
        实现一个三量子比特门，在 `ctl_q = 1` 时会交换两个目标比特，
        但是当两个目标比特都为 `1` 的时候会添加一个 (-1) 的相位。也就是说，
        除了 |111> ---> -|111>，该多量子比特门等同于 Fredkin 门。
        矩阵形式：
            \begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & -1 \\
            \end{pmatrix}
        
        Args:
            ctl_q: Qubit
                控制比特

            targ_qubit1: Qubit
                目标比特1

            targ_qubit2: Qubit
                目标比特2

            zero_ctl: bool
                为 `False` 时，只在控制比特为 1 时会交换两量子比特。
                为 `True` 时，只在控制比特为 0 时会交换两量子比特。
    """
    ctl_phase_swap_circ = QCircuit()

    ctl_phase_swap_circ << CNOT(targ_qubit2, targ_qubit1)

    ctl_phase_swap_circ << RY(targ_qubit2, np.pi/4)       \
                        << CNOT(targ_qubit1, targ_qubit2) \
                        << RY(targ_qubit2, np.pi/4)

    if zero_ctl:
        ctl_phase_swap_circ << X(ctl_q)
    
    ctl_phase_swap_circ << CNOT(ctl_q, targ_qubit2)

    if zero_ctl:
        ctl_phase_swap_circ << X(ctl_q)
    
    ctl_phase_swap_circ << RY(targ_qubit2, -np.pi/4)      \
                        << CNOT(targ_qubit1, targ_qubit2) \
                        << RY(targ_qubit2, -np.pi/4)

    ctl_phase_swap_circ << CNOT(targ_qubit2, targ_qubit1)

    return ctl_phase_swap_circ

def ctl_pswap_block(
    ctl_q: Qubit,
    targ_reg1: QVec,
    targ_reg2: QVec,
    zero_ctl: bool = False
) -> QCircuit:
    """
        对两个相同大小的量子寄存器的对应位置上的量子比特施加 `ctl_pswap`，交换两个量子寄存器，
        并对 |11> 添加 (-1)的相位。

        Args:
            ctl_q: `Qubit`
                控制比特

            targ_reg1: `QVec`
                目标量子寄存器1

            targ_reg2: `QVec`
                目标量子寄存器2

            zero_ctl: `bool`
                为 `False` 时，只在控制比特为 1 时会交换两量子寄存器。
                为 `True` 时，只在控制比特为 0 时会交换两量子寄存器。
    """
    n = len(targ_reg1)

    if len(targ_reg2) != n:
        raise Exception(f"Register sizes not match!, {len(targ_reg1)} != {len(targ_reg2)}")
    
    cpswap_block_circ = QCircuit()

    if zero_ctl:
        cpswap_block_circ << X(ctl_q)

    for i in range(n):
        cpswap_block_circ << ctl_pswap(
            ctl_q,
            targ_reg1[i],
            targ_reg2[i],
            zero_ctl = False
        )

    if zero_ctl:
        cpswap_block_circ << X(ctl_q)
    
    return cpswap_block_circ



def ctl_mod_multiply(
    ctl_q: Qubit,
    z_reg: QVec,
    zero_reg: QVec,
    M: int,
    N: int,
    approx_level: int = 0
) -> QCircuit:
    """
        In-place quantum classical controlled modular multiplication

        In-place 的模乘法量子线路。要求 `M` 与 `N` 互质。完成以下计算。
        |yi>|z>  --(yi=0)--> |yi>|z>, 
        |yi>|z>  --(yi=1)--> |yi>|M*z mod N>

        名称为 "in-place" 是因为作为结果的 `M*z mod N` 覆写到了
        `z` 的量子寄存器上。

        Args:
            ctl_q: `Qubit`
                控制比特

            z_reg: `QVec`
                模乘法的被乘数所在量子寄存器， 量子数据。

            zero_reg: `QVec`
                全置为0的量子寄存器，辅助计算用。

            M: `int`
                模乘法的乘数，经典数据。

            N: `int`
                模乘法的模数，经典数据。

            approx_level: `int`
                用于控制线路中用到的傅立叶变换的近似度。
    """

    n = int(np.ceil(np.log2(N)))
    m = int(np.ceil(np.log2(n)))

    if len(z_reg) != n:
        raise Exception("z_reg size not correct!", len(z_reg), n)
    if len(zero_reg) != (n + m + 1):
        raise Exception("zero-reg size not correct!", len(zero_reg), (n + m + 1))
    
    zero_mp1 = zero_reg[:m+1]
    zero_n = zero_reg[m+1:]

    circ_cMod = QCircuit()

    circ_cMod << ctl_pswap_block(
        ctl_q     = ctl_q,
        targ_reg1 = z_reg,
        targ_reg2 = zero_n,
        zero_ctl  = True
    )

    circ_cMod << OOP_MM(
        z        = z_reg,
        zero_mp1 = zero_mp1,
        zero_n   = zero_n,
        M        = M,
        N        = N,
        approx_level = approx_level
    )

    circ_cMod << ctl_reg_swap(
        ctl_q     = ctl_q,
        targ_reg1 = z_reg,
        targ_reg2 = zero_n,
        zero_ctl  = False
    )

    circ_cMod << OOP_MM(
        z            = z_reg,
        zero_mp1     = zero_mp1,
        zero_n       = zero_n,
        M            = M,
        N            = N,
        reverse      = True,
        inverse_M    = True,
        approx_level = approx_level
    )

    circ_cMod << ctl_pswap_block(
        ctl_q     = ctl_q,
        targ_reg1 = z_reg,
        targ_reg2 = zero_n,
        zero_ctl  = True
    )

    return circ_cMod