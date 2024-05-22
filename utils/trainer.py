from torch.utils.tensorboard import SummaryWriter
import copy
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np_
import pickle
import optax
import jaxopt

jax.config.update("jax_enable_x64", True)

class Trainer(eqx.Module):
    writer: SummaryWriter
    loss: str
    problem:str
    metric: str
    dict: dict()

    def __init__(self, logdir="runs", Loss='mse', metric='mse', problem='vectors'):
        self.writer= SummaryWriter(logdir)
        self.loss=Loss
        self.problem=problem
        self.metric=metric
        self.dict={}
    
    
    #---------------------------------------------- Vectors & matrices
    #------------------------------------------------------------------
    @eqx.filter_jit
    def loss_fn_class(self, params, statics, x, y):
        model = eqx.combine(params, statics)
        pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
        return -jnp.mean(y * pred_y)

    @eqx.filter_jit
    def loss_fn_mse(self, params, statics, x, y):
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model)(x))**2)
    
    @eqx.filter_jit
    def accuracy_vectors(self,params, statics, x, y):
        model = eqx.combine(params, statics)
        pred = jnp.argmax( jax.nn.softmax(jax.vmap(model)(x)), axis=1) 
        y = jnp.argmax( y, axis=1) 
        return jnp.mean(pred == y)
    
    @eqx.filter_jit
    def mse_vectors(self,params, statics,  x, y):
        model = eqx.combine(params, statics)
        return jnp.mean( optax.l2_loss(y, jax.vmap(model)(x)) )


    #------------------------------------------------------------ Graphs 
    #-------------------------------------------------------------------
    @eqx.filter_jit
    def loss_fn_class_graph(self, params, statics, x, y, adj=None):
        model = eqx.combine(params, statics)
        logits = model(x, adj)
        # #print(logits.shape, y.shape)
        # #print("the logits shape", logits.shape)
        return -jnp.mean(y * jax.nn.log_softmax(logits))

    @eqx.filter_jit
    def loss_fn_mse_graph(self, params, statics, x, y, adj=None):
        model = eqx.combine(params, statics)
        return jnp.mean((y - jax.vmap(model(x, adj)))**2)
    
    @eqx.filter_jit
    def accuracy_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return model(x, adj)
    @eqx.filter_jit
    def accuracy_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jnp.mean(jnp.argmax(model(x, adj), axis=1) == jnp.argmax(y, axis=1))
    
    @eqx.filter_jit
    def mse_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jnp.mean((jax.vmap(model)(x, adj) - y)**2 )
    

    # ------------------------------------------------------------ Graphs
    # -------------------------------------------------------------------

    @eqx.filter_jit
    def get_pred(self, params, statics, x):
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x)

    @eqx.filter_jit
    def get_pred_graphs(self, params, statics, x, y, adj):
        model = eqx.combine(params, statics)
        return jax.vmap(model)(x, adj)


    # -------------------------------------------------------------------
    def return_loss_grad(self, params, batch, static):
            if self.problem=='vectors':
                (x, y) = batch
                if self.loss == 'class':
                    grads =jax.grad(self.loss_fn_class)(params, static, x, y)
                    loss = self.loss_fn_class(params, static, x, y)
                elif self.loss=='mse':
                    grads= jax.grad(self.loss_fn_mse)(params, static, x, y)
                    loss  =self.loss_fn_mse(params, static, x, y)
            elif self.problem== 'graphs':
                # #print("I came here and wemnt to calculate the loss")
                (x, y, adj) = batch
                if self.loss == 'class':
                    grads  =jax.grad(self.loss_fn_class_graph)(params, static, x, y, adj=adj)
                    loss = self.loss_fn_class_graph(params, static, x, y, adj=adj)
                elif self.loss=='mse':
                    grads  =jax.grad(self.loss_fn_mse_graph)(params, static, x, y, adj=adj)
                    loss =  self.loss_fn_mse_graph(params, static, x, y, adj=adj)
            return loss, grads
    
    # -------------------------------------------------------------------
    @eqx.filter_jit
    def return_V_star(self, x, params, data):
        statics, y = data
        model = eqx.combine(params, statics)
        # #print("In Vstar -- The problem is", self.problem, "The loss is", self.loss)
        if self.problem=='vectors':
            # #print("The problem is", self.problem, "The loss is", self.loss)
            if self.loss == "class":
                # #print("1The problem is", self.problem, "The loss is", self.loss)
                y=y.astype(jnp.int64)
                pred_y = jax.nn.log_softmax(jax.vmap(model)(x))
                pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
                return -jnp.mean(y * pred_y)
            elif self.loss=="mse":
                # #print("3 -- The problem is", self.problem, "The loss is", self.loss)
                # #print("loss -- The value of the loss function", jnp.mean(optax.l2_loss(y, jax.vmap(model)(x))))
                return jnp.mean(optax.l2_loss(y, jax.vmap(model)(x)))
            
        # #print("I am not going where I am suppsoed to")
    # -------------------------------------------------------------------
    @eqx.filter_jit
    def return_metric(self, x, params, data):
        statics, y = data
        model = eqx.combine(params, statics)
        if self.problem=='vectors':
            if self.metric == 'class':
                y=y.astype(jnp.int64)
                pred_y = jax.vmap(model)(x)
                pred_y = jnp.argmax(pred_y, axis=1)
                return jnp.mean(y == pred_y)
            elif self.metric=='mse':
                return jnp.mean(optax.l2_loss(y, jax.vmap(model)(x)))
            
        # elif self.problem== 'graphs':
        #     # #print("I came here and wemnt to calculate the loss")
        #     (x, y, adj) = batch
        #     if self.loss == 'class':
        #         grads  =jax.grad(self.loss_fn_class_graph)(params, static, x, y, adj=adj)
        #         loss = self.loss_fn_class_graph(params, static, x, y, adj=adj)
        #     elif self.loss=='mse':
        #         grads  =jax.grad(self.loss_fn_mse_graph)(params, static, x, y, adj=adj)
        #         loss =  self.loss_fn_mse_graph(params, static, x, y, adj=adj)
        # return loss
        
        
        # 
    
    # -------------------------------------------------------------------
    @eqx.filter_jit
    def return_Hamiltonian(self, params, data):
        static, (x, y, exp_x, exp_y, deltax, flag) = data
        # #print("x", x.shape, "exp_x", exp_x.shape)
        
        x_tog = jnp.concatenate( [ x, exp_x])
        y_tog = jnp.concatenate( [ y, exp_y])
        
        # #print("x tog", x_tog.shape)        
        # Term 1 
        V_star_max = self.return_V_star(x_tog, params, (static, y_tog))
        
        
        #print(V_star_max)
        # Term 2
        dVstar_dx = jax.grad(self.return_V_star,argnums=(0))(exp_x, params, (static, exp_y))
        # #print("shape", dVstar_dx.shape)
        dVstar_dx_deltax = jnp.sum(jnp.trace( jnp.inner(dVstar_dx, deltax ) ))
        
        
        # Term 3
        delta_theta   = jax.grad(self.return_V_star,argnums=(1) )(x_tog, params, (static, y_tog))
        dVstar_dtheta = jax.grad(self.return_V_star, argnums=(1))(exp_x, params,(static, exp_y))
        dVstar_dtheta = jax.tree_util.tree_leaves(dVstar_dtheta )
        delta_theta   = jax.tree_util.tree_leaves(delta_theta )
        dVstar_dtheta_delta_theta=0.
        
        for (dVstar_dw, delta_w)  in zip(dVstar_dtheta,  delta_theta):
            # #print("the operand", dVstar_dw.shape, delta_w.shape)
            # delta_w = delta_w/jnp.sqrt(jnp.linalg.norm(delta_w)**2)
            if len(delta_w.shape)>1:
                inner = jnp.inner(dVstar_dw, delta_w)
                # #print(inner.shape)
                dVstar_dtheta_delta_theta+=jnp.sum(jnp.trace(inner) )
            else:
                inner = jnp.inner(dVstar_dw, delta_w)
                # #print(inner.shape)
                dVstar_dtheta_delta_theta+=jnp.sum(inner )
        # + +0*+ dVstar_dtheta_delta_theta 
        #+dVstar_dtheta_delta_theta+dVstar_dx_deltax
        
        # #print(dVstar_dx_deltax.shape, dVstar_dtheta_delta_theta.shape, )
        return  (V_star_max+flag[0]*dVstar_dx_deltax+flag[1]*dVstar_dtheta_delta_theta),\
                (V_star_max, dVstar_dx_deltax, dVstar_dtheta_delta_theta)
    
        #        dou_V_star_max_x = jnp.mean(optax.l2_loss(y_tog, jax.vmap(model)(x_tog_)) -
        #     optax.l2_loss(y_tog, jax.vmap(model)(x_tog)) )
        # dou_V_star_max_theta = jnp.mean(optax.l2_loss(y_tog, jax.vmap(model)(x_tog))
        #                   - optax.l2_loss(y_tog, jax.vmap(modelN)(x_tog)))
        # H = V_star_max + dou_V_star_max_x + dou_V_star_max_theta




    # ----------------------------------------------------------------------------
    # ----------------------------------------------------------------------------
    def train__CL__reg(self, trainloader, exploader, valloader, testloader, params,\
                    static, optim_outer, n_iter=1000,\
                    save_iter=10, task_id=0,\
                    config={}, dictum = {}):
        trainiter = iter(trainloader)
        expiter = iter(exploader)
        # optim_inner_x, optim_inner_mod = optim_inner
        batch = next(trainiter)
        x, y = batch
        x = x.numpy().astype(np_.float64)
        y = y.numpy().astype(np_.float64)
        batch = (x, y)
        opt_state = optim_outer.init(params)
        from tqdm import tqdm 
        pbar = tqdm(range(n_iter))
        if task_id>0:
            flag=config["flag"]
        else:
            flag=config["flag"]
        # #print("Now the flag is ", flag)
        # jax.value_and_grad(self.return_loss_function_CL, has_aux=True)
        # grad_loss_fn_inner = jax.value_and_grad(self.return_loss_function_CL_inner)
        # grad_loss_fn_inner_mod = jax.value_and_grad(self.return_loss_function_CL_inner_mod)
        # start_iter_inner = task_id*n_iter*inner_iter
        sum_delta_x =0.
        for step in pbar:
            
            try:
                batch = next(trainiter)
            except StopIteration:
                trainiter = iter(trainloader)
                batch = next(trainiter)
            try:
                batch_ex = next(expiter)
            except StopIteration:
                expiter = iter(exploader)
                batch_ex = next(expiter)

            (x, y) = batch
            (exp_x, exp_y) = batch_ex
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            
            # #print(exp_x.shape, x.shape)
            
            delta_x = jnp.abs(exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)            
            data = static, ( x, y, exp_x, exp_y, delta_x, flag)             
            grad, losses = jax.grad(self.return_Hamiltonian,\
                        argnums=(0), has_aux=True)(params,data)         
            (V_star_max, dVstar_dx, dVstar_dtheta)  = losses         
            grad_leav = jax.tree_util.tree_leaves(grad)
            grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
            updates, opt_state = optim_outer.update(grad, opt_state, params)
            params =  optax.apply_updates(params, updates)
            
            
            
            pbar.set_postfix({"Train/MSE:": V_star_max,
                              "Train/dVstar_dx:": dVstar_dx,
                              "Train/dVstar_dtheta:": dVstar_dtheta,
                              "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
                              "Train/||dH_dtheta||:": grad_norm,
                              "Train/Metric:": grad_norm
                            })
            self.writer.add_scalar('train/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
            self.writer.add_scalar('train/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dVstar_dx',
                                   dVstar_dx.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dH_dtheta',
                                   grad_norm.item(), step+task_id*n_iter)

            
            dictum["train"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                V_star_max+dVstar_dx+dVstar_dtheta,\
                grad_norm, grad_norm )

            ## Validation Metric calculations on the total exp_replay
            if step %100==0:
                sum_delta_x=0.
                V_star_max=[]
                dVstar_dx=[]
                dVstar_dtheta=[]
                H=[]
                loader_1, loader_2= valloader
                for (batch_x, batch_ex) in zip(loader_1, loader_2):
                    (x, y) = batch_x
                    (exp_x, exp_y) = batch_ex
                    x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                
                    delta_x = jnp.abs(exp_x-x)
                    sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
                    delta_x = (delta_x/sum_delta_x)
                    data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
                    h, losses = self.return_Hamiltonian(params,data)         
                    (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
                    V_star_max.append(V_star_max_b)
                    dVstar_dx.append(dVstar_dx_b)
                    dVstar_dtheta.append(dVstar_dtheta_b)
                    H.append(h)
                    
                V_star_max=np_.mean(V_star_max)
                dVstar_dx= np_.mean(dVstar_dx)
                dVstar_dtheta=np_.mean(dVstar_dtheta)
                H=np_.mean(H)
                
                # #print(H,  dVstar_dx, dVstar_dtheta)
                # pbar.set_postfix({"Valid/MSE:": V_star_max,
                #               "Train/dVstar_dx:": dVstar_dx,
                #               "Train/dVstar_dtheta:": dVstar_dtheta,
                #               "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
                #               "Train/||dH_dtheta||:": grad_norm\
                #             })
                
                self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/Loss/MSE', V_star_max.item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
                self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                
                dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                V_star_max+dVstar_dx+dVstar_dtheta )
                
        ## Test Metric calculations on the total exp_replay
        sum_delta_x=0.
        V_star_max=[]
        dVstar_dx=[]
        dVstar_dtheta=[]
        H=[]
        loader_1, loader_2= valloader
        for (batch_x, batch_ex) in zip(loader_1, loader_2):
            (x, y) = batch_x
            (exp_x, exp_y) = batch_ex
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
            delta_x = (exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)
            data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
            h, losses = self.return_Hamiltonian(params,data)         
            (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
            V_star_max.append(V_star_max_b)
            dVstar_dx.append(dVstar_dx_b)
            dVstar_dtheta.append(dVstar_dtheta_b)
            H.append(h)
            
                
                
        V_star_max=np_.mean(V_star_max)
        dVstar_dx= np_.mean(dVstar_dx)
        dVstar_dtheta=np_.mean(dVstar_dtheta)
        H=np_.mean(H)
        self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),task_id)
        self.writer.add_scalar('test/Loss/MSE', V_star_max.item(), task_id)
        self.writer.add_scalar('test/gradient/dVstar_dx',
                            dVstar_dx.item(), task_id)
        self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), task_id)
        dictum["test"+str(task_id)] = (V_star_max,dVstar_dx, dVstar_dtheta, H)
        
        self.writer.flush()
        
        
        return params, static, optim_outer, dictum
    
  
    def train__CL__class(self, trainloader, exploader, valloader, testloader, params,\
                    static, optim_outer, n_iter=1000,\
                    save_iter=10, task_id=0,\
                    config={}, dictum = {}):
        trainiter = iter(trainloader)
        expiter = iter(exploader)
        # optim_inner_x, optim_inner_mod = optim_inner
        batch = next(trainiter)
        x, y = batch
        x = x.numpy().astype(np_.float64)
        y = y.numpy().astype(np_.float64)
        batch = (x, y)
        opt_state = optim_outer.init(params)
        from tqdm import tqdm 
        pbar = tqdm(range(n_iter))
        if task_id>0:
            flag=config["flag"]
        else:
            flag=config["flag"]
        # #print("Now the flag is ", flag)
        # jax.value_and_grad(self.return_loss_function_CL, has_aux=True)
        # grad_loss_fn_inner = jax.value_and_grad(self.return_loss_function_CL_inner)
        # grad_loss_fn_inner_mod = jax.value_and_grad(self.return_loss_function_CL_inner_mod)
        # start_iter_inner = task_id*n_iter*inner_iter
        sum_delta_x =0.
        metrics=0.0
        for step in pbar:
            
            try:
                batch = next(trainiter)
            except StopIteration:
                trainiter = iter(trainloader)
                batch = next(trainiter)
            try:
                batch_ex = next(expiter)
            except StopIteration:
                expiter = iter(exploader)
                batch_ex = next(expiter)

            (x, y) = batch
            (exp_x, exp_y) = batch_ex
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            
            #--------------------------------------------------
            # print("in the train loop", exp_x.shape, x.shape)
            delta_x = jnp.abs(exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)            
            data = static, ( x, y, exp_x, exp_y, delta_x, flag)             
            grad, losses = jax.grad(self.return_Hamiltonian,\
                        argnums=(0), has_aux=True)(params,data)         
            (V_star_max, dVstar_dx, dVstar_dtheta)  = losses         
            grad_leav = jax.tree_util.tree_leaves(grad)
            grad_norm = jnp.sqrt(sum([jnp.linalg.norm(g)**2 for g in grad_leav])/len(grad_leav) )
            updates, opt_state = optim_outer.update(grad, opt_state, params)
            params =  optax.apply_updates(params, updates)
            
            
            # metric+= (1/100)*self.return_metric(x, params, (static, y)  )
            pbar.set_postfix({"Train/Cross Entropy:": V_star_max,
                              "Train/dVstar_dx:": dVstar_dx,
                              "Train/dVstar_dtheta:": dVstar_dtheta,
                              "Train/H:":  V_star_max+dVstar_dx+dVstar_dtheta,
                              "Train/||dH_dtheta||:": grad_norm,
                              "Train/Metric:": metrics
                            })
            self.writer.add_scalar('train/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
            self.writer.add_scalar('train/Loss/cross entropy', V_star_max.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dVstar_dx',
                                   dVstar_dx.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/gradient/dH_dtheta',
                                   grad_norm.item(), step+task_id*n_iter)
            self.writer.add_scalar('train/metric',
                                   metrics, step+task_id*n_iter)
            
            dictum["train"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                V_star_max+dVstar_dx+dVstar_dtheta, metrics)
            
            
            ## Validation Metric calculations on the total exp_replay
            if step %100==0:
                sum_delta_x=0.
                V_star_max=[]
                dVstar_dx=[]
                dVstar_dtheta=[]
                H=[]
                metrics=[]
                loader_1, loader_2= valloader
                for (batch_x, batch_ex) in zip(loader_1, loader_2):
                    (x, y) = batch_x
                    (exp_x, exp_y) = batch_ex
                    x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                    exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
                
                    delta_x = jnp.abs(exp_x-x)
                    sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
                    delta_x = (delta_x/sum_delta_x)
                    data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
                    h, losses = self.return_Hamiltonian(params,data)         
                    (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
                    V_star_max.append(V_star_max_b)
                    dVstar_dx.append(dVstar_dx_b)
                    dVstar_dtheta.append(dVstar_dtheta_b)
                    H.append(h)
                    metrics.append( self.return_metric(exp_x, params,  (static, exp_y)  ) )
                    
                V_star_max=np_.mean(V_star_max)
                dVstar_dx= np_.mean(dVstar_dx)
                dVstar_dtheta=np_.mean(dVstar_dtheta)
                H=np_.mean(H)
                metrics= np_.mean(metrics)
                self.writer.add_scalar('Valid/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/Loss/cross entropy', V_star_max.item(), step+task_id*n_iter)
                self.writer.add_scalar('Valid/gradient/dVstar_dx', dVstar_dx.item(), step+task_id*n_iter)
                self.writer.add_scalar('valid/gradient/dVstar_dtheta', dVstar_dtheta.item(), step+task_id*n_iter)
                self.writer.add_scalar('valid/metrics', metrics, step+task_id*n_iter)
                dictum["valid"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                V_star_max+dVstar_dx+dVstar_dtheta, metrics)
        ## Test Metric calculations on the total exp_replay
        sum_delta_x=0.
        V_star_max=[]
        dVstar_dx=[]
        dVstar_dtheta=[]
        H=[]
        metrics=[]
        loader_1, loader_2= valloader
        for (batch_x, batch_ex) in zip(loader_1, loader_2):
            (x, y) = batch_x
            (exp_x, exp_y) = batch_ex
            x = x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            y = y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_x = exp_x.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
            exp_y = exp_y.numpy().astype(np_.float64)[:min(exp_x.shape[0], x.shape[0])]
        
            delta_x = (exp_x-x)
            sum_delta_x += jnp.sqrt((jnp.linalg.norm(delta_x)**2))
            delta_x = (delta_x/sum_delta_x)
            data = static, ( x, y, exp_x, exp_y, delta_x, flag) 
            h, losses = self.return_Hamiltonian(params,data)         
            (V_star_max_b, dVstar_dx_b, dVstar_dtheta_b)  = losses  
            V_star_max.append(V_star_max_b)
            dVstar_dx.append(dVstar_dx_b)
            dVstar_dtheta.append(dVstar_dtheta_b)
            H.append(h)
            metrics.append( self.return_metric(exp_x, params,  (static, exp_y)  ) )
                
                
        V_star_max=np_.mean(V_star_max)
        dVstar_dx= np_.mean(dVstar_dx)
        dVstar_dtheta=np_.mean(dVstar_dtheta)
        H=np_.mean(H)
        metrics=np_.mean(metrics)
        self.writer.add_scalar('test/Loss/H', (V_star_max+dVstar_dx+dVstar_dtheta).item(),task_id)
        self.writer.add_scalar('test/Loss/cross entropy', V_star_max.item(), task_id)
        self.writer.add_scalar('test/gradient/dVstar_dx',
                            dVstar_dx.item(), task_id)
        self.writer.add_scalar('test/gradient/dVstar_dtheta', dVstar_dtheta.item(), task_id)
        self.writer.add_scalar('test/metric', metrics, task_id)
        dictum["test"+str(step+task_id*n_iter)] = ( V_star_max,dVstar_dx, dVstar_dtheta, \
                V_star_max+dVstar_dx+dVstar_dtheta, metrics)
        
        self.writer.flush()
        return params, static, optim_outer, dictum



    def evaluate__(self, epoch, batch, params, static):
        
        # --- Get Loss
        if self.problem=='vectors':
            (x, y) = batch
            if self.loss == 'class':
                loss = self.loss_fn_class(params, static, x, y)
            elif self.loss=='mse':
                loss, grads = self.return_loss_grad(params, (x,y), static)
                grads = jax.tree_util.tree_leaves(grads)
                grads = jnp.mean(jnp.asarray([jnp.linalg.norm(g) for g in grads]))
            
        elif self.problem== 'graphs':
            (x, y, adj) = batch
            if self.loss == 'class':
                loss  =self.loss_fn_class_graph(params, static, x, y, adj=adj)
            elif self.loss=='mse':
                loss  =self.loss_fn_mse_graph(params, static, x, y, adj=adj)

        # --- Get score
        if self.problem == 'vectors':
            (x, y) = batch
            if self.loss == 'class':
                score =self.accuracy_vectors(params, static, x, y)
            elif self.loss=='mse':
                score =self.mse_vectors(params, static, x, y)
        elif self.problem== 'graphs':
            (x, y, adj) = batch
            if self.loss == 'class':
                score = self.accuracy_graphs(params, static, x, y, adj=adj)
            elif self.loss=='mse':
                score =self.accuracy_graphs(params, static, x, y, adj=adj)
                
        # --- Get prediction
        if self.problem=='vectors':
            (x, _) = batch
            pred= self.get_pred(params, static, x)
        if self.problem == 'graphs':
            (x, _) = batch
            pred = self.get_pred(params, static, x)
                                

        return loss, score, pred, grads





    def writer(self, dict, epoch, string_scalers= ['train'], metric_scaler=['training_loss', 'validation_loss', 'loss', 'acc']):
        for (string, metric) in zip(string_scalers, metric_scaler):
            self.writer.add_scalar(str(string), dict[metric], epoch)
        pickle.dump( dict['params'], open("best_ckpt.pkl"), "wb")







    def train__NONEUC__(self, trainloader, params, static, optim, n_iter=1000):
        # #print(x)
        # #print(y)
        # x = x.numpy().astype(np_.float64)
        # adj = adj.numpy().astype(np_.float64)
        # y = jax.nn.one_hot(jnp.array(y.astype(np_.int32)), 7)
        (x, y, adj) = trainloader
        opt_state = optim.init_state(params, batch=(x, y, adj), static=static)
        loss__ = []
        score__ = []
        from tqdm import tqdm
        pbar = tqdm(range(n_iter))
        for step in pbar:
            (x, y, adj) = trainloader
            params, opt_state = optim. update(
                params=params, state=opt_state,  batch=(x, y, adj), static=static)
            # trainer.train__(step, batch, params, static, optim, opt_state)
            # params , static = eqx.partition(model, eqx.is_array)
            l, s = self.evaluate__(step, (x, y, adj), params, static)
            loss__.append(l)
            score__.append(s)
            pbar.set_postfix({"Loss:": sum(loss__)/len(loss__),
                             "accuracy": sum(score__)/len(score__)})
