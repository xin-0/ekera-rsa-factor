from math import comb, gcd, sqrt
from random import choices, randint
import numpy as np
from pyqpanda import *
from typing import Tuple

from fpylll import *
from fpylll import BKZ

# from pytest import approx
# from sympy import false
from circ_modules import *



class EkeraSolver:

    def __init__(self,
        N: int,
        s: int = 2,
        generator: int = 0,
        approx_level: int = 0
    ) -> None:

        if N%2 == 0:
            raise Exception("N can't be even.")
        
        if s < 2:
            raise Exception("s must be greater than 1")

        self.s = s
        self.N = N

        self.m = int(np.ceil(np.log2(N)/2))
        self.l = int(np.ceil(self.m/s))

        # 初始化 g 和 x
        N_half = (N-1)//2

        if generator == 0:
            self.g = randint(2, N-2)
            self.x = pow(self.g, N_half, N)
            while True:
                self.g = randint(2, N-2)
                self.x = pow(self.g, N_half, self.N)
                if (gcd(self.g, N) == 1) and (gcd(self.x, N) == 1) and self.x != 1:
                    break
        else:
            if (generator == 1) or (generator >= (N-1)):
                raise Exception(f"generator must be in range [2, N-2]")
            else:
                self.g = generator
                self.x = pow(self.g, N_half, self.N)
        
        print(f"g = {self.g}, x = {self.x}")
        
        M_list_a = [self.g]

        ##### 利用经典计算提前算出来

        for i in range(1, self.m + self.l):
            next_M_a = (M_list_a[i-1] * M_list_a[i-1]) % self.N
            M_list_a.append(next_M_a)

        # x 与 N 互质，由初始化过程确保
        M_list_b = [pow(self.x, -1, self.N)]

        for i in range(1, self.l):
            next_M_b = (M_list_b[i-1] * M_list_b[i-1]) % self.N
            M_list_b.append(next_M_b)
        

        self.qvm = CPUQVM()
        self.qvm.init_qvm()

        self.mod_circ_n = int(np.ceil(np.log2(self.N)))
        self.mod_circ_m = int(np.ceil(np.log2(self.mod_circ_n)))

        a = self.qvm.qAlloc_many(self.m + self.l)
        b = self.qvm.qAlloc_many(self.l)
        z = self.qvm.qAlloc_many(self.mod_circ_n)
        zero = self.qvm.qAlloc_many(self.mod_circ_n + self.mod_circ_m + 1)

        self.__a_meas = self.qvm.cAlloc_many(self.m + self.l)
        self.__b_meas = self.qvm.cAlloc_many(self.l)
        self.__z_meas = self.qvm.cAlloc_many(self.mod_circ_n)

        self.__isConstructed = False

        self.__circuit = QCircuit()

        self.__circuit << self.__constructCirc(
            a = a,
            b = b,
            z = z,
            zero = zero,
            M_a = M_list_a,
            M_b = M_list_b,
            approx_level = approx_level
        )

        self.__prog = QProg()

        self.__prog << self.__circuit

        self.__prog << measure_all(a, self.__a_meas) \
                  << measure_all(b, self.__b_meas) \
                  << measure_all(z, self.__z_meas)

        self.__prog_result = None
        self.__j_k_pairs = None

        return


    def __constructCirc(
        self,
        a: QVec,
        b: QVec,
        z: QVec,
        zero:QVec,
        M_a: List[int],
        M_b: List[int],
        approx_level: int = 0
    ) -> QCircuit:
        """
            搭建Ekerå量子线路
        """
        

        ekera_circ = QCircuit()

        # 让 a,b 寄存器进入叠加态
        ekera_circ << apply_QGate(a, H) << apply_QGate(b, H)
        ekera_circ << init_circ(z, '1')
        # 计算 g^a (modN)
        for i, m_a in enumerate(M_a):
            ekera_circ << ctl_mod_multiply(
                ctl_q        = a[i],
                z_reg        = z,
                zero_reg     = zero,
                M            = m_a,
                N            = self.N,
                approx_level = approx_level
            )

         # 计算 x^(-b) (modN)
        for i, m_b in enumerate(M_b):
            ekera_circ << ctl_mod_multiply(
                ctl_q        = b[i],
                z_reg        = z,
                zero_reg     = zero,
                M            = m_b,
                N            = self.N,
                approx_level = approx_level
            )

        # 对 a,b 寄存器进行傅立叶变换
        ekera_circ << QFT(a) << QFT(b)

        self.__isConstructed = True

        return ekera_circ

    def __defineLattice(
        self,
        list_jk_n: List[Tuple]
    ):
        """
            L = 

              j1       j2       ...     js       1
            2^(l+m)     0       ...      0       0
               0      2^(l+m)   ...      0       0
               .        .                .       .
               .        .                .       .
               0        0       ...    2^(l+m)   0  

            v =[{-2^m * k1} mod 2^(l+m),
                {-2^m * k2} mod 2^(l+m),
                            .
                            .
                            .
                {-2^m * kn} mod 2^(l+m),
                            0          ]
        """

        # construct lattice
        num_pairs = len(list_jk_n)
        self.__L = np.zeros((num_pairs+1, num_pairs+1))
        self.__L[0, num_pairs] = 1

        two_lm = 2**(self.l+self.m)
        for i in range(num_pairs):
            self.__L[0, i] = list_jk_n[i][0]
            self.__L[i+1,i] = two_lm
        
        self.__L = self.__L.astype(int)

        self.__v = []

        for i in range(num_pairs):
            vi = ((-1) * 2**(self.m) * list_jk_n[i][1]) % two_lm
            if vi >= two_lm//2:
                vi = vi - two_lm
            self.__v.append(vi)
        self.__v.append(0)

        return self.__L, self.__v

    def runProg(
        self,
        num_it: int = 0
    ):
        """
            运行量子程序，获得测量结果并取到(j,k)对
        """
        if num_it == 0:
            num_it = self.s + 1

        result = self.qvm.run_with_configuration(
            self.__prog, 
            self.__a_meas[:] + self.__b_meas[:] + self.__z_meas[:], 
            num_it)
        
        self.__prog_result = result

        self.__j_k_pairs = []

        for meas_res in result:
            z_res = int(meas_res[:self.mod_circ_n], 2)
            k = int(meas_res[self.mod_circ_n : self.mod_circ_n + self.l], 2)
            j = int(meas_res[self.mod_circ_n + self.l:], 2)
            # 互不相同的 (j,k)
            if (j,k) not in self.__j_k_pairs:
                self.__j_k_pairs.append((j,k))
        
        return 

    def getResult(self):
        """
            取测量结果
        """
        if self.__prog_result == None:
            raise Exception("Prog have not run yet!")
        
        return self.__prog_result
    
    def getJKPairs(self):
        """
            取(j,k)对
        """
        if self.__prog_result == None:
            raise Exception("Prog have not run yet!")
        
        return self.__j_k_pairs
    

    def getCircuit(self):
        """
            取Ekerå量子线路
        """
        # if not self.__isConstructed:
        #     raise Exception("Need to construct circuit first!")
        
        return self.__circuit
    
    def getQProg(self):
        # if not self.__isConstructed:
        #     raise Exception("Need to construct circuit first!")
        
        return self.__prog

    def getIteration(self):

        return self.__post_iteration


    def postProcess(
        self,
        pair_list: List[Tuple] = None
    ) -> bool:
        """
            args:
                pair_list: `List[Tuple]`
                    进行经典后处理所使用的jk对
            
            return:
                bool:
                    后处理是否成功分解 N
        """
        self.__post_iteration = 0

        if pair_list == None:
            pair_list = self.__j_k_pairs

        fail_thresh = comb(len(pair_list), self.s + 1)

        while True:
            self.__post_iteration += 1
            # 在所有测量中随机取 (s+1) 个 (j,k) 对
            jk_n = choices(pair_list, k = self.s + 1)
            # 构建 Lattice
            L, v = self.__defineLattice(jk_n)
            A = IntegerMatrix.from_matrix(L.astype(int).tolist())
            # Lattice basis reduction
            L_bkz = BKZ.reduction(A, BKZ.Param(self.s+1))
            M = GSO.Mat(L_bkz)
            M.update_gso()
            # 找最近的lattice vector
            u_coef = M.babai(v)
            u = L_bkz.multiply_left(u_coef)
            # 如果找到 d 退出循环
            if pow(self.g, u[-1], self.N) == self.x and u[-1]>1:
                self.d = u[-1]
                break
            # 循环次数超出排列组合数量，退出循环，认为没找到d
            # 可能是因为不满足 d < r，可尝试重新重新构建  EkeraSolver
            if self.__post_iteration > fail_thresh:
                return False
        
        # 已找到 d, 开始后处理找 p、 q
        c = self.d + 1
        
        delt = int(sqrt(c*c - self.N))
        self.p = c + delt
        self.q = c - delt
        # 确认找到
        if self.p * self.q != self.N:
            return False

        return True
