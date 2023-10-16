% function [outputArg1,outputArg2] = visualize(type_in, psize_in, nr_in, met_in, org_in, file)

close all
% handles = findall(groot, 'Type', 'figure', 'Tag', 'volshow');
% close(handles);
% clear all

%%% For fast Data output into excel table


% type_in;       % 1,2 -> "Erosion", "Dilation"
% psize_in;      % [1, 4]   -> 2, 3, 4, 5
% nr_in;         % [1, 4]   -> "DLDP_081", "DLDP_082", "DLDP_088", "DLDP_090"
% met_in;        % [1, 4]   -> "Max", "Mean", "DMax", "DMean"
% org_in;        % [1, 10]  -> tv, bs, hl, hr, el, er, chiasm, opnl, opnr, brain
% file           % 1,2 -> CTV, PTV


% for t = 1:2
%     for n = 1:4
%         for m = 1:4
%             for o = 1:9
%                 writeEx(1, 1, n, m, o, 1);
%             end
%         end
%     end
% end

visualize(1, 2, 1, 2, 1, 1);



%%% For PTV already evaluated: E3 D3
%%% For CTV already evaluated: E3 D3 E2



