function DCETS = dce(TS, m, tau)

DCETS = zeros(length(TS) - tau*(m-1), m);
for ii = 1:length(DCETS)
    for jj = 1:m
        DCETS(ii,jj) = TS((ii)+(m-jj)*tau);
    end
end