U_length # U_length: (int)
V_length # V_length: (int)
U_seq # U_seq: (torch tensor) [seq_len, feature size dim]
V_seq # V_seq: (torch tensor) [seq_len, feature size dim]

def softmax_tcc(left, right, j):
    right_length = right.size(0)
    top = torch.exp(-1 * torch.dist(left, right[j])**2)
    down_inner = []                
    for k in range(right_length):
        down_inner.append(torch.exp(-1 * torch.dist(left, right[k])**2))
    down = torch.sum(torch.stack(down_inner))
    return top / down

# tcc uses l2 distance: dist_l2(a,b)=||a-b|| = torch.sqrt(torch.sum((a_i-b_i)**2)
for i in range(U_length):
    v_bar_list = []
    for j in range(V_length):
        if i == j:
            continue
        # a_j = torch.softmax(-1*torch.dist(U_seq[i],V_seq[j])**2, 0) # not working
        a_j = softmax_tcc(U_seq[i], V_seq, j)
        v_bar_j = a_j * V_seq[j]
        v_bar_list.append(v_bar_j)

    v_bar = torch.sum(torch.stack(v_bar_list),0)

    x_k_list = []
    for k in range(U_length):
        x_k = -1 * torch.dist(v_bar, U_seq[k])**2 # l2 distance
        x_k_list.append(x_k)
    x = torch.stack(x_k_list)

    y = torch.zeros(U_length).cuda()
    y[i] = torch.tensor(1)
    y_bar = torch.softmax(x, 0)
    L_cbc = -1 * torch.sum(y*torch.log(y_bar))

    B_k_list = []
    for k in range(U_length):
        # B_k = torch.softmax(-1 * torch.dist(v_bar,U_seq[k])**2, 0) # not working
        B_k = softmax_tcc(v_bar, U_seq, k)
        B_k_list.append(B_k)
    B = torch.stack(B_k_list)                

    mu = torch.tensor(0.).cuda()
    for k in range(U_length):
        mu += B[k] * k

    sigma_pow = torch.tensor(0.).cuda()
    for k in range(U_length):
        sigma_pow += B[k] * (k - mu)**2

    ramda = 0.01 # regularization weight
    L_cbr = (i-mu)**2/sigma_pow + ramda*torch.log(torch.sqrt(sigma_pow))
