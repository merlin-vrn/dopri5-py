from bisect import bisect

class integrator():
    """Выполняет численное интегрирование задачи Коши вырабнным методом.
    
    Хранит состояние интегрирования уравнения.
    
    Объект является callable, принимает параметр — координату, до которой нужно интегрировать"""
    
    def __init__(self, method_step, method_order, method_dense, func, params, x0, t0, t1, tol, dt0 = None):
        self.f = func
        self._method_step = method_step # метод должен быть вложенным или эмулировать поведение вложенного метода, оценивая погрешность шага
        self._method_order = method_order
        self._method_dense = method_dense # задать None, если метод не поддерживает плотную выдачу
        self.p = params
        self.x = x0
        self.t = t0
        self.t0 = t0
        self.tol = tol
        self.dt = dt0 if dt0 else self._calculate_first_step_size(t1) # если величина шага не задана, прикинуть по алгоритму
        self.solution = {} # запоминать все результаты рассчётов в виде словаря t: (x, err, dt, ki)
        self.coordinate = [] # координаты в упорядоченном виде; OrderedDict потребовал бы извлекать этот список каждый раз комадной list(d)
        self.steps_rejected = 0 # количество отброшенных шагов из-за недостаточной точности
    
    def __call__(self, t, use_dense=True):
        """Выдать значение интегрируемой функции в t. Если требуется, интегрировать выбранным методом до t
        
        Если use_dense=True, можно выдать значение в указанной точке с помощью плотной выдачи (не требуется, чтобы интегратор проходил через эту точку)
        Если use_dense=False, и интегратор "перешагнул" через точку t, необходимо вычислить значение, сделав шаг точно в эту точку"""
        
        assert t>=self.t0, f"Интегрирование начато в точке {self.t0}, было запрошено значение в {t}. Обратное интегрирование не поддерживается"
        while t>=self.t: # интегрировать вперёд до тех пор, пока мы не пройдём требуемую точку
            self._step()
        if t==self.t:
            return self.x
        if t in self.solution: # есть значение точно в этой точке
            return self.solution[t][0] 
        # ищем подходящий отрезок 
        idx = bisect(self.coordinate, t) # idx пригодится, если потребуется вставлять новое решение
        coordinate = self.coordinate[idx - 1]
        solution = self.solution[coordinate]
        if use_dense and self._method_dense: # используем плотную выдачу только если метод поддерживает
            return self._method_dense(solution[0], solution[3], (t - coordinate)/solution[2]) # solution[2] = dt, solution[3] = ki
        else: #if use_dense and self._method_dense
            r = self._method_step(self.f, solution[0], coordinate, self.p, t-coordinate) # solution[0] = x
            # TODO нужно ли сохранять результат интегрирования, как _step()? Для сохранения порядка его нужно вставить в позицию idx в self.coordinate
            return r[0]

    def _step(self):
        """Выполняет один шаг выбранным методом. Использует технику управления длиной шага (ХНВ I, стр. 177, 178)
        Используется di = 1 в формуле 4.6, т.е. погрешность считается абсолютной"""
        
        # TODO: сделать это параметризуемым. Также, реализовать рекомендацию сбрасывать facmax в 1 для шагов после отброшенных
        fac = 0.9 # гарантийный множитель
        facmax = 5 # максимальная скорость роста шага 
        while True:
            xn, err, ki = self._method_step(self.f, self.x, self.t, self.p, self.dt)
            dtn = self.dt * min(facmax, fac * (self.tol/err)**(1/(self._method_order+1)) ) # в любом случае регулируем шаг
            if err < self.tol:
                # шаг принят
                break
            # шаг не принят
            self.dt = dtn
            self.steps_rejected += 1
        self.coordinate.append(self.t)
        self.solution[self.t] = (self.x, err, self.dt, ki)
        self.t += self.dt
        self.x = xn
        self.dt = dtn
        
    def _calculate_first_step_size(self, t1):
        """Автомтический выбор первого шага (ХНВ I, стр. 194, 195)
        
        Возвращает эффективное значение первого шага по алгоритму в книге."""
        
        ff = f(self.x, self.t, self.p)
        # FIXME не очень понятно, что в книге понимается под x0 и xmax.
        # На тестируемой задаче второе слагаемое значительно больше первого, поэтому это не сказывается.
        den = max(abs(self.t), abs(t1))**(-self._method_order-1) + sum(i*i for i in ff)**((self._method_order+1)/2)
        dt1 = (self.tol/den)**(1/(self._method_order+1))
        x1 = [y+dt1*gg for y,gg in zip(self.x,ff)] # один шаг методом Эйлера
        ff = f(x1, self.t+dt1, self.p)
        den = max(abs(self.t+dt1), abs(t1+dt1))**(-self._method_order-1) + sum(i*i for i in ff)**((self._method_order+1)/2)
        dt2 = (self.tol/den)**(1/(self._method_order+1))
        return min(dt1, dt2)
        
    def __str__(self):
        return f"<integrator: x={self.x}, t={self.t}, dt={self.dt}; {len(self.coordinate)} points>"

    def __repr__(self):
        return f"<integrator: x={self.x}, t={self.t}, dt={self.dt}; {len(self.coordinate)} points>"

def dopri5_step(f, x, t, p, dt):
    """Делает шаг методом Домрана-Принса 5(4) (ХНВ I, стр. 182)
    
    Возвращает новое значение интегрируемой функции в точке t+dt, погрешность и
    набор промежуточных коэффициентов (стадий), которые можно использовать для
    интерполяции промежуточных значений с помощью формул плотной выдачи,
    реализованной в функции dopri5_dense
    
    f - функция правой части задачи Коши
    x - текущее значение, из которого делаем шаг
    t - текущая координата
    p - параметры функции f
    dt - размер шага
    
    Коэффициенты взяты из таблицы 4.6
    
    Примечание: ki здесь соответствуют произведениям ki*h в книге"""
    
    # TODO: реализовать оптимизацию, сэкономить одно вычисление функции
    # за счёт переиспользования k7 с предыдущего шага: то, что там было z, сюда передано как x; то, что там было равно t+dt, сюда передано как t,
    # с учётом того, что ki здесь соответствуют ki*dt в книге, а dt с предыдущего шага изменился; поэтому здесь k1 = k7_prev*dt_new/dt_prev (ПРОВЕРИТЬ)
    k1 = [g*dt for g in f(x, t, p)]
    k2 = [g*dt for g in f([y+l1/5 for y,l1 in zip(x, k1)], t+dt/5, p)]
    k3 = [g*dt for g in f([y+(l1*3+l2*9)/40 for y,l1,l2 in zip(x,k1,k2)], t+dt*3/10, p)]
    k4 = [g*dt for g in f([y+l1*44/45-l2*56/15+l3*32/9 for y,l1,l2,l3 in zip(x,k1,k2,k3)], t+dt*4/5, p)]
    k5 = [g*dt for g in f([y+l1*19372/6561-l2*25360/2187+l3*64448/6561-l4*212/729 for y,l1,l2,l3,l4 in zip(x,k1,k2,k3,k4)], t+dt*8/9, p)]
    k6 = [g*dt for g in f([y+l1*9017/3168-l2*355/33+l3*46732/5247+l4*49/176-l5*5103/18656 for y,l1,l2,l3,l4,l5 in zip(x,k1,k2,k3,k4,k5)], t+dt, p)]
    z  = [y+l1*35/384+l3*500/1113+l4*125/192-l5*2187/6784+l6*11/84 for y,l1,l2,l3,l4,l5,l6 in zip(x,k1,k2,k3,k4,k5,k6)]
    k7 = [g*dt for g in f(z, t+dt, p)]
    z1 = [y + l1*5179/57600+l3*7571/16695+l4*393/640-l5*92097/339200+l6*187/2100+l7/40 for y,l1,l2,l3,l4,l5,l6,l7 in zip(x,k1,k2,k3,k4,k5,k6,k7)]
    err = sum([abs(y-y1) for y,y1 in zip(z,z1)])
    return (z, err, (k1, k2, k3, k4, k5, k6, k7))

def dopri5_dense(x, ki, th):
    """Плотное решение порядка 4 формулы Дормана-Принса 5(4) (ХНВ I, стр. 191)
    
    вычисляет значение "между шагами" метода, x(t + th * dt)
    
    x - значение в точке, из которой делали шаг методом Дормана-Принса, т.е. x(t)
    ki - рассчитанные на этом шаге промежуточные значения (стадии) метода, выданные для этого шага функцией dopri5_step
    th - интерполяционный параметр, доля шага, которую нужно выполнить, 0≤th≤1,
        th=0 - вычислит x, т.е. x(t); bi окажутся равными нулю
        th=1 - вычислит результат полного шага, т.е. x(t + dt); bi в этом случае окажутся равны коэффициентам в строке "z" функции dopri5_step
    
    Примечание: ki и bi здесь соответствуют произведениям ki*h и bi*h в формулах (5.7)"""
    
    th2 = th*th
    th3 = th2*th
    th4 = th2*th2
    b1 = th - th2*1337/480  + th3*1039/360   - th4*1163/1152
    b3 =      th2*4216/1113 - th3*18728/3339 + th4*7580/3339
    b4 =    - th2*27/16     + th3*9/2        - th4*415/192
    b5 =    - th2*2187/8480 + th3*2673/2120  - th4*8991/6784
    b6 =      th2*33/35     - th3*319/105    + th4*187/84
    return [y + l1*b1+l3*b3+l4*b4+l5*b5+l6*b6 for y,l1,l2,l3,l4,l5,l6,l7 in zip(*((x,)+ki))]

def f(x, t, p):
    # p = (s, r, b)
    y0 = p[0]*(x[1]-x[0])
    y1 = (p[1]-x[2])*x[0]-x[1]
    y2 = x[0]*x[1]-p[2]*x[2]
    return (y0, y1, y2)

def test():
    problem = integrator(dopri5_step, 5, dopri5_dense, f, (10, 28, 8/3), x0=(1,1,1), t0=0, t1=100, tol=1E-10)
    print(problem(20))
    print(f"{len(problem.coordinate)} точек, {problem.steps_rejected} шагов отброшено")
    
def chop(i, e):
    return round(i/e)*e

def poincare(runs, gran, b):
    s = b+1+(2*(b+1)*(b+2))**(1/2)
    r = s*(s+b+3)/(s-b-1)
    
    zc = r-1
    print(f"params: s={s:11.8f}, r={r:11.8f}, b={b:11.8f}; critical plane: z={zc:11.8f}")
    
    l = (1,1,1)
    is_over = l[2]>zc
    t = 0

    p = integrator(dopri5_step, 5, dopri5_dense, f, (s, r, b), x0=l, t0=t, t1=100, tol=1E-12)
    
    period = {}
    t_period_found = None
    t_period = 0
    
    for i in range(runs):
        t_prev = t
        t = i/gran
        l_prev = l
        l = p(t)
        is_over_prev = is_over
        is_over = l[2]>zc
        if (is_over != is_over_prev) and is_over_prev: # ищем только "протыкание вниз",
#        if is_over != is_over_prev:
            # отловили пересечение, сходимся к нему
            q = '^' if is_over else 'v'
            
            tl = t_prev
            tr = t

            while abs(tl-tr)>1E-10:
                tc = (tl + tr)/2
                if (p(tc)[2]>zc) == is_over:
                    tr = tc
                else:
                    tl = tc
                    
            lc = p(tc)
            lc_chop = (chop(lc[0],1E-7), chop(lc[1],1E-7))
            if lc_chop in period:
                t_period_found = period[lc_chop]
                t_period = tc-t_period_found
#                print(f"t={tl:11.8}: x={lc[0]:11.8f}, y={lc[1]:11.8f} ({q}) again; dt = {t_period}")
                break
            else:
                period[lc_chop] = tc
#            print(f"t={tl:11.8}: x={lc[0]:11.8f}, y={lc[1]:11.8f} ({q})")
    if t_period_found:
        period = [k for k,t in period.items() if t>=t_period_found]
        print(f"limit cycle on poincare section after t={float(t_period_found):.8}, period {float(t_period):.8}, len={len(period)}")
        return (len(period), t_period_found, t_period)
    else:
        print("no period found")
        return None

def scan_doubling():
    b = 0.13
    db = 0.002 # после нахождения третьего удвоения периода нужно сокращать db каждый раз в отношение из разностей * гарантийный множитель 0.1
    dob = []
    dis = []
    period_len = 2
    while True:
        b_prev = b
        period_len_prev = period_len

        b += db
        pl, tpf, tp = poincare(100000, 10, b)
        period_len = pl
        
        if period_len_prev != period_len:
            if period_len == period_len_prev*2:
                print (f"DOUBLING occured between {b_prev} and {b}")
            else:
                b -= db
                period_len = period_len_prev
                db = db/2
                print (f"CHAOS around {b}! stepping back, new db={db}!")
                continue
            # нашли удвоение, сходимся к нему
            bl = b_prev
            br = b
            while br-bl>1E-6:
                bc = (bl+br)/2
                pl, tpf, tp = poincare(100000, 10, bc)
                if pl == period_len:
                    br = bc
                else:
                    bl = bc

            b = br # далее ищем из заведомо "правой" точки, но максимально близкой к точке удвоения
            db = db/2
            print(f"FOUND doubling at {bc}, new db={db}")
            
            if db<1E-5: # большую точность мы обеспечить пока что не можем
                break
            
def scan_doubling_result()
    d=(
        (4,   0.13700097656250002),
        (8,   0.14193750000000002),
        (16,  0.14383007812500004),
        (32,  0.14411621093750004),
        (64,  0.14417675781250003),
        (128, 0.14418945312500003),
        )
    print ([(q3[1]-q2[1])/(q2[1]-q1[1]) for q1,q2,q3 in zip(d[:-2],d[1:-1],d[2:])])

if __name__=='__main__':
        
#    poincare(100000, 10, 0.1335)
#    scan_doubling()
    
