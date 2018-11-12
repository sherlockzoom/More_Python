# 剑指offfer编程题(python版)
标签（空格分隔）： python

---
[toc]

### 动态规划

#### 和为s的两个数字
```py
# 依次遍历第一个找到的两个数就是乘积最小的一个
# 设置两个指针，初始为第一个low和最后一个hight， low只能加， hight只能减
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        if not array:  # 数组为空返回[]
            return []
        low = 0
        hight = len(array)-1
        while low < hight:
            tmp_sum = array[low]+array[hight]
            if tmp_sum > tsum: # 当前和大于tsum，那么需要减值
                hight -= 1
            elif tmp_sum < tsum:
                low += 1
            else:
                return [array[low], array[hight]]
        return []
```
```py
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
	    # 和为s的两个数字
        ret = []
        for i, num in enumerate(array):
            tmp = tsum - num
            if tmp in array[i+1:]:
                ret.append((num*tmp, num, tmp))
        if not ret:
            return ret
        tmp_ret = min(ret) #默认(num*tmp, num, tmp) num*tmp作为关键码求最小
        return tmp_ret[1:]
```
#### 和为s的连续正数序列
> 考虑两个数ｓｍａｌｌ和ｂｉｇ分别表示当前最小值和最大值。初始设置为１，２．如果从ｓｍａｌｌ到ｂｉｇ序列的和大于ｓ，我们可以从序列中去掉较小值，否则增加ｂｉｇ值。

```py
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        if tsum < 3:  # 验证至少2个数
            return []
        small, big = 1, 2
        middle = (1+tsum)>>1  # 最大值--终止条件
        cur_sum = small + big
        ret = []
        while small < middle:
            if cur_sum == tsum:
                ret.append(range(small, big+1))
            while cur_sum > tsum and small < middle: # 当前和大于tsum，减小small
                cur_sum -= small
                small += 1
                if cur_sum == tsum:
                    ret.append(range(small, big+1))
            big += 1
            cur_sum += big
        return ret
```
#### 连续子数组和最大
```py
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if not array: # 数组为空返回0
            return 0
        dp = [float('-inf')]  # 初始值负无穷
        for i,n in enumerate(array):
            if dp[i] <= 0:   # dp[i]前面最大的连续数组和，如果小于等于0，那么加上当前值只会更小，更新dp[i+1]=n
                dp.append(n)
            else:
                dp.append(dp[i]+n)  # 当前值为0，且前面连续最大和为正，说明加上当前数一定大于之前和
        return max(dp)
```

### 数组与矩阵

#### 构建乘法数组

> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素`B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]`。不能使用除法。

```py
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        first = [1]
        second = [1]
        for i in range(1, len(A)):
            first.append(first[i-1]*A[i-1]) # 依次保存中间的计算值
            second.append(second[i-1]*A[-i])
        B = []
        for i in range(0, len(A)):
            B.append(first[i]*second[-(i+1)])
        return B
            
```

#### 不用加减乘除做加法

```py
# -*- coding:utf-8 -*-
class Solution:
    def Add(self, num1, num2):
        # 使用sum函数
        return sum([num1, num2])
```

#### 孩子们的游戏（约瑟夫环）

> 1） 开始有n个人，从0数到m-1，m-1下标的这个人出列，下一次有n-1个人，从0数到m-1，m-1下标出列，。。下一个出列的下标 = （当前下标 + m-1）%当前人数


```py
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        if n < 1 or m < 1:
            return -1
        last = 0
        people = range(n)
        i = 0
        for num in range(n,1,-1):
            i = (i+m-1)%num
            people.pop(i)
        return people[-1]
```

#### 扑克牌顺子

```py
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        if not numbers or len(numbers) != 5:
            return False
        zeros = numbers.count(0)
        gap = 0
        i,j = zeros, zeros+1
        n = len(numbers)
        numbers.sort()
        while j < n:
            if numbers[i]==numbers[j]:
                return False
            gap += numbers[j] - numbers[i] - 1
            i = j
            j += 1
        return True if gap <= zeros else False
        # write code here
```

#### 数组中只出现一次的数字

```py
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        ret = []
        for num in array:
            if array.count(num)==1:
                ret.append(num)
        if len(ret)==2:
            return ret
        return [0,0]
        # write code here
```

#### 数字在排序数组中出现的次数

```py
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # o(n)解法,笔试编程题一般没问题，想必面试是过不了的，利用数组有序，做二分查找
        return data.count(k)

# log(n)的解法，二分查找在找到ｋ的时候做变形使得满足要求
# -*- coding:utf-8 -*-
class Solution:
    def bisect_first_k(self, data, k, start, end):
        # 找到第一个ｋ
        if start > end:
            return -1
        mid = (start+end)>>1
        if data[mid]==k:
            if mid > 0 and data[mid-1]!=k or mid==0:
                return mid
            else:
                end = mid -1
        elif data[mid] > k:
            end = mid - 1
        else:
            start = mid + 1
        return self.bisect_first_k(data, k, start, end)
    
    def bisect_last_k(self, data, k, start, end):
        # 找到最后一个ｋ
        if start > end:
            return -1
        mid = (start+end)>>1
        if data[mid]==k:
            if mid < len(data)-1 and data[mid + 1] != k or mid ==len(data)-1:
                return mid
            else:
                start = mid + 1
        elif data[mid] > k:
            end = mid - 1
        else:
            start = mid + 1
        return self.bisect_last_k(data, k , start, end)
    
    def GetNumberOfK(self, data, k):
        count = 0
        if data and len(data)>0:
            first = self.bisect_first_k(data, k, 0, len(data)-1)
            last = self.bisect_last_k(data, k, 0, len(data)-1)
            if first > -1 and last > -1:
                return last - first + 1
        return count
        
        # write code here
        
```

#### 丑数

> 把只包含因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

```py
class Solution:
    def GetUglyNumber_Solution(self, index):
        # write code here
        n, m = 0, 0
        k1, k2, k3 = [1], [1], [1]
        while n < index:
            m = min(k1[0], k2[0], k3[0])
            n += 1
            if m in k1:
                k1.remove(m)
            if m in k2:
                k2.remove(m)
            if m in k3:
                k3.remove(m)
            k1.append(m*2)
            k2.append(m*3)
            k3.append(m*5)
        return m
```

#### 把数组排成最小的数

> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

```py
# -*- coding:utf-8 -*-
class Solution:
    def str_cmp(self, s1, s2):  #　定义排序比较规则
        s1, s2 = s1+s2, s2+s1
        return cmp(s1, s2)

    def PrintMinNumber(self, numbers):
        if not numbers:
            return ''
        tmp = map(str, numbers) #　转化为ｓｔｒ
        tmp.sort(self.str_cmp)
        return ''.join(tmp)
```

#### 整数中１出现的次数

> 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数。

```py
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        count = 0
        for i in range(1,n+1):
            count += str(i).count('1')
        return count
```

#### 最小的ｋ个数

> [python中堆的使用][1]

[python-coolbook-查找最大或最小的ｋ个值][2]

```py
# -*- coding:utf-8 -*-
import heapq
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if tinput==None or len(tinput)<k:
            return []
        return heapq.nsmallest(k, tinput)
        # write code here
```


#### 数组中出现次数超过一半的数字


```py
# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        mid = len(numbers)>>1
        for num in numbers:
            if numbers.count(num) > mid:  # py中的ｃｏｕｎｔ（）使用
                return num
        return 0
```

#### 顺时针打印矩阵

> 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每个数字

```py
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        if not matrix:
            return None
        rows = len(matrix)
        cols = len(matrix[0])
        start = 0
        result = []
        while rows > 2*start and cols > 2*start:
            endx = rows - 1 - start  
            endy = cols - 1 - start
            for i in range(start, endy+1):  # 左到右处理
                result.append(matrix[start][i])
            if start < endx:　　# 上到下处理
                for i in range(start+1,endx+1):
                    result.append(matrix[i][endy])
            if start < endx and start < endy:　　＃　右到左处理
                for i in range(endy-1, start-1, -1):
                    result.append(matrix[endx][i])
            if start < endx-1 and start < endy:　　＃　下到上处理
                for i in range(endx-1, start, -1):
                    result.append(matrix[i][start])
            start += 1
        return result
```

#### 旋转数组的最小数字

```py
# -*- coding:utf-8 -*-
# 所有元素都是大于0， 所以零pre=0
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        if len(rotateArray)==0:
            return 0
        pre = 0
        for num in rotateArray:
            if num < pre:
                return num
            pre = num
        return rotateArray[0] 
        #  有序，输出第一个元素
```


### 字符串

>　python处理字符串实在是便利。

#### 正则表达式匹配

> 实现`.` `*`　的匹配

```py
# -*- coding:utf-8 -*-

def match_core(s, pat):
    if len(s)==0 and len(pat)==0:  #匹配完成返回True
        return True
    if len(s)>0 and len(pat)==0: #匹配串pat匹配完，而s不为空，说明不匹配
        return False
    if len(pat)>1 and pat[1]=='*':  # pat至少有2个*匹配才有意义
        if len(s)>0 and (s[0]==pat[0] or pat[0]=='.'):  # ab  .*  / ab a*  两种情况统一处理
            return match_core(s[1:], pat) or match_core(s, pat[2:]) or match_core(s[1:], pat[2:])  # *匹配1个保持当前模式/不匹配/移动到下一个状态
        else:  # len(s)==0 s匹配完，看剩下的是否匹配
            return match_core(s, pat[2:])
    if len(s)>0 and (pat[0]=='.' or pat[0]==s[0]):  # 处理非匹配字符和 . 直接处理下一个
        return match_core(s[1:], pat[1:])
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        return match_core(s, pattern)
        # write code here
```

#### 字符串转整数

```py

# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        if not s:
            return 0
        str2num = {'1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '0':0}
        flag2num = {'-':-1, '+': 1}
        first = s[0]
        if first in ['+', '-']: # 包含符号位的情况
            flag = flag2num[first]
            tmp = 0
            for n in s[1:]:
                if n not in str2num:
                    return 0
                tmp = tmp*10 + str2num[n]
            return tmp*flag    
        else:
            tmp = 0
            for n in s:
                if n not in str2num:
                    return 0
                tmp = tmp*10 + str2num[n]
            return tmp
```

#### 翻转单词顺序序列

```py
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        return ' '.join(s.split(' ')[::-1])
        # write code here
```

#### 左旋转字符串

```py
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        return s[n:] + s[:n]
        # write code here
```

#### 第一个只出现一次的字符

```py
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        if not s:
            return -1
        for i, ch in enumerate(s):
            if s.count(ch)==1:
                return i
        # write code here
```


#### 字符串的排列

> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。 结果请按字母顺序输出。

```py

# -*- coding:utf-8 -*-
import itertools
class Solution:
    def Permutation(self, ss):
        # 借助内置的itertools.permutaions求解
        # write code here
        if not ss:
            return ss
        result = []
        k = itertools.permutations(ss)
        for i in k:
            result.append(''.join(i))
        result = list(set(result))
        result.sort()
        return result


# -*- coding:utf-8 -*-
def permuations(n):
    #　全排列的一个实现(二找，一交换，一翻转)
    """
    1. 找到排列中最右一个升序的首位置ｉ，　ｘ＝ai
    2. 找到排列中第ｉ位向右最后一个比ａｉ大的位置，ｊ，　ｙ＝ａｊ
    3. 交换ｘ　ｙ
    4. 把第ｉ＋１位到最后的部分翻转
    ２１５４３－－下一个数是２３１４５
    """
    indices = range(n)
    # n = len(ss)
    yield indices
    while True:
        low_idx = n-1
        while low_idx > 0 and indices[low_idx-1] > indices[low_idx]:　# 找到排列中最右一个升序的首位置
            low_idx -= 1
        if low_idx == 0:
            break
        low_idx -= 1
        high_idx = low_idx + 1
        while high_idx < n and indices[high_idx] > indices[low_idx]:　　#找到排列中第ｉ为向右最右一个比ａｉ大的位置
            high_idx += 1
        high_idx -= 1
        indices[low_idx], indices[high_idx] = indices[high_idx], indices[low_idx]　# 交换
        indices[low_idx+1:] = reversed(indices[low_idx+1:])  #　翻转
        yield indices

class Solution:
    def Permutation(self, ss):
        if not ss:
            return []
        ret_set = set()
        for idx in permuations(len(ss)):
            e = ''.join([ss[i] for i in idx])
            ret_set.add(e)
        return sorted(ret_set)
        # write code here
```

#### 替换空格
```py
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        return s.replace(' ', '%20')
```

### 栈和队列

#### 栈的压入、弹出序列
```py
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        n = len(pushV)
        if not n: return False
        tmp = []
        j = 0
        for val in pushV:
            tmp.append(val)  # 依次入栈
            while j < n and tmp[-1]==popV[j]:  # tmp栈顶值等于popV[j]值 出栈
                tmp.pop()
                j += 1
        if tmp:return False
        return True
```

#### 包含ｍｉｎ函数的栈

> 定义栈的数据结构，实现一个能够得到栈最小元素的ｍｉｎ函数

```py
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []　# 存放数据
        self.stack_min = [] #　存放最小值
    def push(self, node):
        if not self.stack_min:
            self.stack_min.append(node)
        else:
            if self.min() <= node:
                self.stack_min.append(self.min())
            else:
                self.stack_min.append(node)
        self.stack.append(node)
            
        # write code here
    def pop(self):
        if not self.stack:
            return []
        else:
            self.stack_min.pop()
            return self.stack.pop()
        # write code here
    def top(self):
        if not self.stack:
            return []
        return self.stack[-1]
        # write code here
    def min(self):
        if not self.stack_min:
            return []
        else:
            return self.stack_min[-1]
        # write code here
```


### 链表

#### 删除链表中重复的结点

```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        if pHead.next.val==pHead.val:
            pnode = pHead.next.next
            while pnode and pnode.val == pHead.val:
                pnode = pnode.next
            return self.deleteDuplication(pnode)
        else:
            # pnode = pHead.next
            pHead.next = self.deleteDuplication(pHead.next)
            return pHead
                
                
        # write code here
```

#### 链表中环的入口结点

```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
def meeting_node(pHead):
    if not pHead:
        return None
    pslow = pHead.next
    if not pslow:
        return None
    pfast = pslow.next
    while pslow and pfast:
        if pslow==pfast:
            return pslow
        pslow = pslow.next
        pfast = pfast.next.next
    return None  # 前面没有相遇返回None

class Solution:
    def EntryNodeOfLoop(self, pHead):
        """
        1. 找到环
        2. 计算环中结点个数
        3. 根据环结点个数设置快指针初始
        4. 快慢指针相遇
        """
        pmeet = meeting_node(pHead)
        if not pmeet:
            return None
        node_count = 1 #当前相遇点包含在环中 1
        pnode = pmeet
        while pnode.next != pmeet:  # 遍历一圈计数得到环中结点个数
            pnode = pnode.next
            node_count += 1
        pnode1 = pHead
        for i in range(node_count):  # pnode1 先走node_count
            pnode1 = pnode1.next
        pnode2 = pHead
        while pnode2!=pnode1:  # pnode2 走node_count步之后指向环的入口
            pnode2 = pnode2.next
            pnode1 = pnode1.next
        return pnode1
        # write code here
```

#### 两个链表的第一个公共结点
```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def iter_node(root):
    while root:
        yield root
        root = root.next

class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None
        p1 = [node for node in iter_node(pHead1)]
        p2 = [node for node in iter_node(pHead2)]
        ret = None
        while p1 and p2:
            top1 = p1.pop()
            top2 = p2.pop()
            if top1.val == top2.val:
                ret = top1
                continue
            else:
                break
        return ret
                
               
#－－－－－－－－－－－－－－－－－－－－

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def iter_node(root):
    while root:
        yield root
        root = root.next

class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 or not pHead2:
            return None
        p1 = [node.val for node in iter_node(pHead1)]
        for node in iter_node(pHead2):
            if node.val in p1:
                return node
        return None
```

#### 二叉搜索树与双向链表

> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        # 中序遍历
        if not pRootOfTree or (not pRootOfTree.left and not pRootOfTree.right): # 只有一个根结点或空时
            return pRootOfTree
        self.Convert(pRootOfTree.left) #　递归处理左子树
        lt = pRootOfTree.left
        if lt:
            while lt.right:
                lt = lt.right
            pRootOfTree.left, lt.right = lt, pRootOfTree　# 修改当前根结点的指针
        
        self.Convert(pRootOfTree.right)　　# 处理右子树
        rt = pRootOfTree.right
        if rt:
            while rt.left:
                rt = rt.left
            pRootOfTree.right, rt.left = rt, pRootOfTree
        
        while pRootOfTree.left: #　最左的结点是双链表的第一个结点
            pRootOfTree = pRootOfTree.left
        return pRootOfTree
        
        # write code here
```
#### 复杂链表复制

> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

```py
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

# 使用生成器来高效访问
def iter_node(root):
    while root:
        yield root
        root = root.next

class Solution:

    # 返回 RandomListNode
    def Clone(self, pHead):
        mem = dict()  # 保存对象内存地址与下标的对应关系
        for i, node in enumerate(iter_node(pHead)):
            mem[id(node)]=i  # id获取对象的内存地址
        lst = [RandomListNode(node.label) for node in iter_node(pHead)]　# 创建ｌｓｔ保存结点值
        
        for p, node in zip(iter_node(pHead), lst): #　复制ｎｅｘｔ和ｒａｎｄｏｍ指向
            if p.next:
                node.next = lst[mem[id(p.next)]]
            if p.random: 
                node.random = lst[mem[id(p.random)]]
        return lst[0] if lst else None
```

#### 合并两个排序的链表

```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        if not pHead1: # pHead1 为空返回ｐＨｅａｄ２
            return pHead2
        elif not pHead2:　＃　pHead2 为空返回pHead1
            return pHead1
        else:  # 都不为空，处理第一个元素
            if pHead1.val <= pHead2.val:
                p = pHead1
                pHead1 = pHead1.next
            else:
                p = pHead2
                pHead2 = pHead2.next
        pnode = p  
        while pHead1 and pHead2:  # 依次处理两个链表
            if pHead1.val <= pHead2.val:
                pnode.next = pHead1
                pnode = pHead1
                pHead1 = pHead1.next
            else:
                pnode.next = pHead2
                pnode = pHead2
                pHead2 = pHead2.next
        ＃处理剩余结点
        if pHead1:
            pnode.next = pHead1
        if pHead2:
            pnode.next = pHead2
        return p
```

#### 反转链表

```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    # 非递归版本
    def ReverseList(self, pHead):
        if not pHead or not pHead.next: # 空或者只有一个元素直接返回
            return pHead
        q = None
        p = pHead
        while p:
            tmp = p.next #暂存下一个结点
            p.next = q # 修改当前结点指向
            q = p # 指向返回链表的第一个元素
            p = tmp # 访问下一个
        return q
        # write code here

#　递归版本
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        else:
            pnode = self.ReverseList(pHead.next) # pnode新表头
            pHead.next.next = pHead # 新表头最后一个结点指向ｐｈｅａｄ
            pHead.next = None　# ｐｈｅａｄ指向None,修改尾指针
            return pnode
        # write code here
        
```
#### 链表中倒数第k个结点

```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 设置间隔为ｋ的两个指针
class Solution:
    def FindKthToTail(self, head, k):
    # 链表k可能大于链表长度，此时返回None
        i = 0
        p = head
        while p and i<k:
            p = p.next
            i += 1
      
        if i==k:  # k小于等于链表长度，正常处理
            q = head
            while p:
                q = q.next
                p = p.next
            return q
        return None
        # write code here
```

#### 从尾到头打印链表
```py
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
from collections import deque
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        if not listNode:
            return []
        tmp = deque()  #　使用队列
        while listNode:
            tmp.appendleft(listNode.val)
            listNode = listNode.next
        return tmp
        # write code here
```


### 递归和循环

#### 矩阵覆盖
> 矩阵覆盖类似与斐波拉契数列

```py
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        a = [0, 1, 2] 
        if number<3:
            return a[number]
        for i in xrange(3, number+1):
            a.append(a[i-1]+a[i-2])
        return a[number]
```

#### 变态跳台阶 
> 动手推导一下：２^(n-1)
```py
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        return 2**(number-1)
       
```
#### 跳台阶

```py
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        a = [0,1,2] # 起步不一样
        if number<3:
            return a[number]
        for i in xrange(3, number+1):
            a.append(a[i-1]+a[i-2])
        return a[number]
        # write code here
```
#### 斐波拉契数列
```py
# -*- coding:utf-8 -*-
# 使用记忆体函数，保存中间值。
from functools import wraps

def memo(func):
    cache = {}
    @wraps(func)
    def wrap(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrap

class Solution:
    @memo
    def Fibonacci(self, n):
        if n==0:
            return 0
        if n<2:
            return 1
        return self.Fibonacci(n-1) + self.Fibonacci(n-2)
       
# 使用list缓存中间值
class Solution_2:
    def Fibonacci(self, n):
        a = [0, 1, 1]
        if n<3:
            return a[n]
        for i in range(3,n+1):
            a.append(a[i-1]+a[i-2])
        return a[n]
        
```

### 树

#### 二叉搜索树的第k个结点

> 给定一颗二叉搜索树，请找出其中的第k小的结点。例如， 5 / \ 3 7 /\ /\ 2 4 6 8 中，按结点数值大小顺序第三个结点的值为4。

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def inorder(pRoot):
    # 中序遍历，只需要找到第k个，不必全部遍历，考虑使用生成器延迟计算
    if pRoot:
        if pRoot.left:
            for lt in inorder(pRoot.left):
                yield lt
        yield pRoot
        if pRoot.right:
            for rt in inorder(pRoot.right):
                yield rt
                

class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        if k < 1 or not pRoot:
            return None
        for i, node in enumerate(inorder(pRoot)):
            if i==k-1:
                return node
        return None
        # write code here
```

#### 序列化二叉树

> 请实现两个函数，分别用来序列化和反序列化二叉树

```py

```

#### 把二叉树打印成多行

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if not pRoot:
            return []
        nodes = [pRoot]
        ret = []
        while nodes:
            cur, nxt = [], [] # 保存每一层结点
            for node in nodes:
                cur.append(node.val)
                if node.left:
                    nxt.append(node.left)
                if node.right:
                    nxt.append(node.right)
            nodes = nxt
            ret.append(cur)
        return ret
        # write code here
```

#### 按之字形顺序打印二叉树

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        if not pRoot:
            return []
        nodes = [pRoot]
        flag = True
        ret = []
        while nodes:
            cur, nxt = [], [] #　保存每一层的结点
            for node in nodes:
                cur.append(node.val)
                if node.left:
                    nxt.append(node.left)
                if node.right:
                    nxt.append(node.right)
            nodes = nxt
            if flag:
                ret.append(cur)
                flag = False
            else:
                ret.append(cur[::-1])
                flag = True
        return ret
        # write code here
```

#### 对称的二叉树

> 请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def help_func(self, pRoot1, pRoot2):
        if not pRoot1 and not pRoot2:
            return True
        if not pRoot1 or not pRoot2:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.help_func(pRoot1.left, pRoot2.right) and self.help_func(pRoot1.right, pRoot2.left)
    
    def isSymmetrical(self, pRoot):
        return self.help_func(pRoot, pRoot)
        # write code here
```

#### 二叉树的下一个结点

> 考察中序遍历

```py
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None  # 指向父节点
class Solution:
    def GetNext(self, pNode):
        # ｐｎｏｄｅ右子树最左的点或者ｐｎｏｄｅ父节点
        if not pNode:
            return None
        elif pNode.right:
            prt = pNode.right
            while prt.left:
                prt = prt.left
            return prt
        elif pNode.next:            
            parent = pNode.next
            while parent and parent.left!=pNode:
                parent = parent.next
                pNode = pNode.next
            return parent if parent else None

        # write code here
```

#### 平衡二叉树

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def depth(self, root):
        if not root:
            return 0
        lt = self.depth(root.left)
        rt = self.depth(root.right)
        return max(lt, rt) + 1
    
    def IsBalanced_Solution(self, pRoot):
    # 递归解法，出现多次遍历同一个结点的情况
        if not pRoot:
            return True
        lt = self.depth(pRoot.left)
        rt = self.depth(pRoot.right)
        if abs(lt-rt) > 1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
        # write code here
```

#### 二叉树深度

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
    # 递归解
        if not pRoot:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1
        # write code here
        
# 非递归解
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Depth(self, root):
        
        #　层次遍历得到深度
        if not root:
            return 0        
        from collections import deque
        dp = deque()
        layer = 1
        dp.append((root,1))
        while dp:
            node, layer = dp.popleft()
            deep = layer
            if node.left:
                dp.append((node.left, layer+1))
            if node.right:
                dp.append((node.right, layer+1))
        return deep
        
    def TreeDepth(self, pRoot):
        return self.Depth(pRoot)

```

#### 二叉树中和为某一值的路径

> 输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。**路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径**。

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        if not root:  #　空树　返回空
            return []
        elif root.val > expectNumber: #　当前值大于期望，不存在满足条件路径返回空
            return []
        elif root.val == expectNumber and not root.left and not root.right: #　没有左右子树，当前值等于和，那么返回该结点(即叶子结点)
            return [[root.val]]
        ret = []
        if root.left: ＃　递归处理左子树
            lt = self.FindPath(root.left, expectNumber - root.val)
            for i in lt:
                i.insert(0, root.val)
                ret.append(i)
        if root.right:　# 递归处理右子树
            rt = self.FindPath(root.right, expectNumber - root.val)
            for i in rt:
                i.insert(0, root.val)
                ret.append(i)
        return ret
        # write code here
```

#### 二叉搜索树的后序遍历序列

> 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

```py
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # 二叉搜索树，左子树值小于根，根小于右子树
        #　后序遍历，左－右－根
        # 找到根结点，根据搜索树的特点划分左右子树
        if not sequence or len(sequence)==0:
            return False
        lenght = len(sequence)
        root = sequence[-1]
        breakindex = 0
        while sequence[breakindex] < root and breakindex < lenght-1:
            breakindex += 1
        for i in range(breakindex,lenght):
            if sequence[i] < root:
                return False
        left = True
        if breakindex > 0:
            left = self.VerifySquenceOfBST(sequence[:breakindex])
        right = True
        if breakindex < lenght - 1:
            right = self.VerifySquenceOfBST(sequence[breakindex:lenght-1])
        return left and right
```

#### 从上往下打印二叉树

> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。层次遍历即可

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    # 使用队列
    def PrintFromTopToBottom(self, root):
        if not root:
            return []
        node_queue = deque()
        node_queue.append(root)
        ret = []
        while node_queue:
            first_node = node_queue.popleft() # 队首出队
            ret.append(first_node.val)
            if first_node.left:
                node_queue.append(first_node.left)
            if first_node.right:
                node_queue.append(first_node.right)
        return ret


# 使用ｌｉｓｔ
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        if not root:
            return []
        node_lst, ret = [root], []
        while node_lst:
            ret.append(node_lst[0].val)
            p = node_lst.pop(0)
            if p.left:
                node_lst.append(p.left)
            if p.right:
                node_lst.append(p.right)
        return ret
        
```

#### 二叉树的镜像

> 1) 空树返回空　２）层次遍历入队，栈出队。依次交换当前结点的左右子树

> 递归处理，交换当前结点左右子树，递归处理左右子树

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    # 非递归解
    def Mirror(self, root):
        if not root:
            return None
        node_lst = [root]
        while node_lst:
            node = node_lst.pop()　#　默认弹出最后一个
            node.left, node.right = node.right, node.left
            if node.left:
                node_lst.append(node.left)
            if node.right:
                node_lst.append(node.right)
        return root
        
# 递归解
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # root is None
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.Mirror(root.left)
        self.Mirror(root.right)
        return root

```

#### 树的子结构

> 查找与根结点值相等的结点，依次判断左右子树是否包含同样结构

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def DoesTree1HasTree2(self, pRoot1, pRoot2):
        if not pRoot2:　# pRoot2遍历完，说明包含子结构
            return True
        if not pRoot1: # pRoot1遍历完，而pRoot2不为空
            return False
        if pRoot1.val!=pRoot2.val:
            return False
        return self.DoesTree1HasTree2(pRoot1.left, pRoot2.left) and self.DoesTree1HasTree2(pRoot1.right, pRoot2.right)　# 递归处理左右子结构
    
        
    def HasSubtree(self, pRoot1, pRoot2):
        if not pRoot1 or not pRoot2:　# 空不是子结构
            return False
        result = False
        if pRoot1.val == pRoot2.val:  #当前结点值相等，判断是否包含子结构
            result = self.DoesTree1HasTree2(pRoot1, pRoot2)　# 
        if not result: #　遍历左子树
            result = self.HasSubtree(pRoot1.left, pRoot2)　# 
        if not result:　# 遍历右子树
            result = self.HasSubtree(pRoot1.right, pRoot2)
        return result

```

#### 重建二叉树：给出前序和中序

```py
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        if not pre or not tin:
            return None
        root = TreeNode(pre[0]) #　构造根结点
        idx = tin.index(pre[0])
        left = self.reConstructBinaryTree(pre[1:idx+1], tin[:idx]) # 递归处理左子树
        right = self.reConstructBinaryTree(pre[idx+1:], tin[idx+1:])　# 递归处理右子树
        if left:
            root.left = left
        if right:
            root.right = right
        return root
        # write code here
```

### 数值运算

#### 求１＋２＋３＋...+n(不使用乘除for/while/if/else/switch/case)
```py
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
       	# return (pow(n,2)+n)>>1  利用公式求解
        # return n and self.Sum_Solution(n-1)+n  利用递归求解,注意终止条件
        # return sum(range(1,n+1))  内建公式求解
        return (pow(n,2)+n)>>1
```


#### 调整数组顺序使奇数位于偶数前面

```py
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        num1 = []
        num2 = []
        for num in array:
            if num&0x1==0: # 利用位运算判断奇偶
                num1.append(num)
            else:
                num2.append(num)
        return num2+num1
```

#### 数值的整数次方

> 指数可能正也可能负

```py
# -*- coding:utf-8 -*-
 
class Solution:
    def Power(self, base, exponent):
        if exponent == 0:
            return 1
        if exponent == 1:
            return base
        exp = abs(exponent)
        result = self.Power(base, exp>>1)  # 处理exp/2的情况
        result *= result
        if (exp & 0x1 == 1): # 最后一位是1还需要* base 奇数个base的情况
            result *= base
        if exponent > 0:
            return result
        return 1/result
```

#### 二进制中1的个数
```py
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1(self, n):
        # 内建bin转换为二进制统计1的个数,注意处理正负号的情况
        if n==0:
            return 0
        if n>0:
            return bin(n).count('1')
        else:
            return bin(n&0xffffffff).count('1')
```


  [1]: https://github.com/qiwsir/algorithm/blob/master/heapq.md
  [2]: http://python3-cookbook.readthedocs.io/zh_CN/latest/c01/p04_find_largest_or_smallest_n_items.html
