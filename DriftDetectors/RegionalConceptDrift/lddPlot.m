function lddPlot(b1, b2, t1, outName, xtick, ytick, axisValue, sub4Value)

    figSize = [200 200 1500 300];

    samplingTime = 500;
    epsilon = 50;
    rndSampleSize = 250;

    ldd1 = zeros(size(b1, 1), samplingTime);
    ldd2 = zeros(size(b1, 1), samplingTime);
    for i=1:samplingTime
        b2rndSample=b2(randsample(size(b2, 1),rndSampleSize),:);
        t1rndSample=t1(randsample(size(t1, 1),rndSampleSize),:);
        ldd1(:, i) = getLDDbyKNN(b2rndSample, b1, epsilon);
        ldd2(:, i) = getLDDbyKNN(t1rndSample, b1, epsilon);
    end
    figSize = [200 200 1500 300];
    hFig=figure;
    set(hFig, 'Position', figSize);

    sub1 = subplot(1, 4, 1);
    grid on;
    %grid minor;
    box on;
    title('(a) Train Set & Test Set Distirubtion','FontWeight','Normal','FontSize',14);
    xlabel('x_{1}');
    ylabel('P(X)');
    hold on;

    [f,xi] = ksdensity(b1(:,1), 'function', 'pdf');
    plot(xi,f,'-b','LineWidth',1.5)
    
    [f,xi] = ksdensity(t1(:,1), 'function', 'pdf');
    plot(xi,f,'--r','LineWidth',1.5)
    lg = legend('train data','test data');
    set(lg, 'FontSize', 12 ,'FontWeight','Normal');
    hold off;

    sub2 = subplot(1, 4, 2);
    hold on;
    grid on;
    %grid minor;
    box on;
    title('(b) Train Set & Test Set','FontWeight','Normal','FontSize',14);
    
    xlabel('x_{1}');
    ylabel('x_{2}');
    
    scatter(b1(:,1), b1(:,2), '.b');
    scatter(t1(:,1), t1(:,2), '+r');
    lg= legend('train data','test data');
    set(lg, 'FontSize', 12 ,'FontWeight','Normal');
    hold off;


    ldd1Mean = mean(ldd1, 2);
    ldd2Mean = mean(ldd2, 2);

    ldd1Std = std(ldd1, 0, 2);
    Pthreshold = zeros(size(ldd1Mean, 1), 1);
    Pthreshold = Pthreshold +0.01;
    minX = norminv(Pthreshold, ldd1Mean, ldd1Std);
    %minX = 0;

    % P = zeros(size(ldd1Mean,1), 1);
    % group = {'b', 't'};
    % for i = 1:size(ldd1Mean,1)
    %     
    %     y=[ldd1(i,:)', ldd2(i,:)'];
    %     P(i)=anova1(y, group, 'off');
    % 
    % end
    % 
    % decreasePoints = find(ldd2Mean < ldd1Mean & P < 0.01);
    % increasePoints = find(ldd2Mean >= ldd1Mean | P >= 0.01);

    decreasePoints = find(ldd2Mean < minX);
    minusOne = find(ldd2Mean == -1);
    decreasePoints = [decreasePoints;minusOne];
    increasePoints = find(ldd2Mean >= minX & ldd2Mean > -1);

    sub3 = subplot(1, 4, 3);
    hold on;
    grid on;
    %grid minor;
    box on;

    title('(c) Drift Instances','FontWeight','Normal','FontSize',14);
    xlabel('x_{1}');
    ylabel('x_{2}');
    scatter(b1(decreasePoints,1), b1(decreasePoints,2), '.b');
    scatter(b1(increasePoints,1), b1(increasePoints,2), 'xr');
    lg = legend('train data density-dec point','train data density-inc point');
    set(lg, 'FontSize', 12,'FontWeight','Normal');
    hold off;


    sub4 = subplot(1, 4, 4);
    hold on;
    grid on;
    %grid minor;
    box on;
    title('(d) LDD','FontWeight','Normal','FontSize',14);
    xlabel('x_{1}');
    ylabel('LDD');
    scatter(b1(decreasePoints,1), ldd2Mean(decreasePoints,:), '.b');
    scatter(b1(increasePoints,1), ldd2Mean(increasePoints,:), 'xr');
    lg = legend('train data density-dec point','train data density-inc point');
    set(lg, 'FontSize', 12,'FontWeight','Normal');
    hold off;

    set(sub1, 'Position', [0.05, 0.18, 0.20, 0.7]);
    
    set(sub2, 'Position', [0.29, 0.18, 0.20, 0.7]);
    set(sub2,'XTick',xtick);
    set(sub2,'YTick',ytick);
    set(sub3, 'Position', [0.53, 0.18, 0.20, 0.7]);
    set(sub3,'XTick',xtick);
    set(sub3,'YTick',ytick);
 
    set(sub4, 'Position', [0.77, 0.18, 0.20, 0.7]);
    axis([sub2 sub3], axisValue);
    %axis([sub4], sub4Value);
    
    set(gcf, 'Units', 'centimeters');
    pos = get(gcf,'Position');
    set(gcf, 'PaperSize', [pos(3) pos(4)]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 pos(3) pos(4)]);
    print(outName,'-depsc', '-loose');


end