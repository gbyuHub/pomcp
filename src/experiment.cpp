#include "experiment.h"
#include "boost/timer.hpp"

using namespace std;

EXPERIMENT::PARAMS::PARAMS()
:   NumRuns(1000),
    NumSteps(100000),
    SimSteps(1000),
    TimeOut(3600),
    MinDoubles(0),
    MaxDoubles(20),
    TransformDoubles(-4),
    TransformAttempts(1000),
    Accuracy(0.01),
    UndiscountedHorizon(1000),
    AutoExploration(true)
{
}

EXPERIMENT::EXPERIMENT(const SIMULATOR& real,
    const SIMULATOR& simulator, const string& outputFile,
    EXPERIMENT::PARAMS& expParams, MCTS::PARAMS& searchParams)
:   Real(real),
    Simulator(simulator),
    OutputFile(outputFile.c_str()),
    ExpParams(expParams),
    SearchParams(searchParams)
{
    if (ExpParams.AutoExploration)
    {
        if (SearchParams.UseRave)
            SearchParams.ExplorationConstant = 0;
        else
            SearchParams.ExplorationConstant = simulator.GetRewardRange();
    }
    MCTS::InitFastUCB(SearchParams.ExplorationConstant);
}

void EXPERIMENT::Run()
{
    boost::timer timer;

    MCTS mcts(Simulator, SearchParams);

    double undiscountedReturn = 0.0;
    double discountedReturn = 0.0;
    double discount = 1.0;
    double cumulative_past_rew = 0.0;
    bool terminal = false;
    bool outOfParticles = false;
    int t;
    bool is_rocksample_problem = (Real.GetClassName() == "ROCKSAMPLE" ? true : false);
    // variables only for rocksample problem
    int collected_good_rocks_num = 0;
    int collected_bad_rocks_num = 0;
    int num_check_action = 0;

    STATE* state = Real.CreateStartState();
    if (SearchParams.Verbose >= 1)
    {
        Real.DisplayState(*state, cout);
    }
    if (is_rocksample_problem) {
        ROCKSAMPLE_STATE* rocksample_state = safe_cast<ROCKSAMPLE_STATE*> (state);
        cout << "Number of bad rocks in true environment: " << rocksample_state->num_bad_rocks << endl;
    }

    for (t = 0; t < ExpParams.NumSteps; t++)
    {
        int observation;
        double reward;
        int action = mcts.SelectAction(cumulative_past_rew);
        terminal = Real.Step(*state, action, observation, reward);
        cumulative_past_rew += reward;

        if (is_rocksample_problem) {
            if (action > 4) num_check_action++;
            if (reward < 0) collected_bad_rocks_num++;
            else if (!terminal && reward > 0) collected_good_rocks_num++; 
        }

        Results.Reward.Add(reward);
        undiscountedReturn += reward;
        discountedReturn += reward * discount;
        discount *= Real.GetDiscount();

        if (SearchParams.Verbose >= 1)
        {
            Real.DisplayAction(action, cout);
            Real.DisplayState(*state, cout);
            Real.DisplayObservation(*state, observation, cout);
            Real.DisplayReward(reward, cout);
        }

        if (terminal)
        {
            cout << "Terminated" << endl;
            break;
        }
        outOfParticles = !mcts.Update(action, observation, reward);
        if (outOfParticles)
            break;

        if (timer.elapsed() > ExpParams.TimeOut)
        {
            cout << "Timed out after " << t << " steps in "
                << Results.Time.GetTotal() << "seconds" << endl;
            break;
        }
    }

    if (outOfParticles)
    {
        cout << "Out of particles, finishing episode with SelectRandom" << endl;
        HISTORY history = mcts.GetHistory();
        while (++t < ExpParams.NumSteps)
        {
            int observation;
            double reward;

            // This passes real state into simulator!
            // SelectRandom must only use fully observable state
            // to avoid "cheating"
            int action = Simulator.SelectRandom(*state, history, mcts.GetStatus());
            terminal = Real.Step(*state, action, observation, reward);

            if (is_rocksample_problem) {
                if (action > 4) num_check_action++;
                if (reward < 0) collected_bad_rocks_num++;
                else if (!terminal && reward > 0) collected_good_rocks_num++; 
            }

            Results.Reward.Add(reward);
            undiscountedReturn += reward;
            discountedReturn += reward * discount;
            discount *= Real.GetDiscount();

            if (SearchParams.Verbose >= 1)
            {
                Real.DisplayAction(action, cout);
                Real.DisplayState(*state, cout);
                Real.DisplayObservation(*state, observation, cout);
                Real.DisplayReward(reward, cout);
            }

            if (terminal)
            {
                cout << "Terminated" << endl;
                break;
            }

            history.Add(action, observation);
        }
    }

    Results.Time.Add(timer.elapsed());
    Results.UndiscountedReturn.Add(undiscountedReturn);
    Results.DiscountedReturn.Add(discountedReturn);
    Results.CollectedBadRocks.Add(collected_bad_rocks_num);
    Results.CollectedGoodRocks.Add(collected_good_rocks_num);
    Results.NumCheckAction.Add(num_check_action);
    cout << "Discounted return = " << discountedReturn
        << ", average = " << Results.DiscountedReturn.GetMean() << endl;
    cout << "Undiscounted return = " << undiscountedReturn
        << ", average = " << Results.UndiscountedReturn.GetMean() << endl;
    cout << "Number of bad rocks collected = " << collected_bad_rocks_num << endl;
    cout << "Number of good rocks collected = " << collected_good_rocks_num << endl;
    cout << "Number of check actions = " << num_check_action << endl;
    cout << "Number of steps = " << t << endl;
}

void EXPERIMENT::MultiRun()
{
    for (int n = 0; n < ExpParams.NumRuns; n++)
    {
        cout << "Starting run " << n + 1 << " with "
            << SearchParams.NumSimulations << " simulations... " << endl;
        Run();
        if (Results.Time.GetTotal() > ExpParams.TimeOut)
        {
            cout << "Timed out after " << n << " runs in "
                << Results.Time.GetTotal() << "seconds" << endl;
            break;
        }
    }
}

void EXPERIMENT::DiscountedReturn()
{
    cout << "Main runs" << endl;
    OutputFile << "Simulations\tRuns\tUndiscounted return\tUndiscounted error\tDiscounted return\tDiscounted error\tTime\n";

    SearchParams.MaxDepth = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);
    ExpParams.SimSteps = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);
    ExpParams.NumSteps = Real.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);

    for (int i = ExpParams.MinDoubles; i <= ExpParams.MaxDoubles; i++)
    {
        SearchParams.NumSimulations = 1 << i;
        SearchParams.NumStartStates = 1 << i;
        if (i + ExpParams.TransformDoubles >= 0)
            SearchParams.NumTransforms = 1 << (i + ExpParams.TransformDoubles);
        else
            SearchParams.NumTransforms = 1;
        SearchParams.MaxAttempts = SearchParams.NumTransforms * ExpParams.TransformAttempts;

        Results.Clear();
        MultiRun();

        cout << "Simulations = " << SearchParams.NumSimulations << endl
            << "Runs = " << Results.Time.GetCount() << endl
            << "Undiscounted return = " << Results.UndiscountedReturn.GetMean()
            << " +- " << Results.UndiscountedReturn.GetStdErr() << endl
            << "Discounted return = " << Results.DiscountedReturn.GetMean()
            << " +- " << Results.DiscountedReturn.GetStdErr() << endl
            << "Time = " << Results.Time.GetMean() << endl
            << "Collected bad rocks = " << Results.CollectedBadRocks.GetMean()
            << " +- " << Results.CollectedBadRocks.GetStdErr() << endl
            << "Collected good rocks = " << Results.CollectedGoodRocks.GetMean()
            << " +- " << Results.CollectedGoodRocks.GetStdErr() << endl
            << "Apply check actions = " << Results.NumCheckAction.GetMean()
            << " +- " << Results.NumCheckAction.GetStdErr() << endl;
        OutputFile << SearchParams.NumSimulations << "\t"
            << Results.Time.GetCount() << "\t"
            << Results.UndiscountedReturn.GetMean() << "\t"
            << Results.UndiscountedReturn.GetStdErr() << "\t"
            << Results.DiscountedReturn.GetMean() << "\t"
            << Results.DiscountedReturn.GetStdErr() << "\t"
            << Results.Time.GetMean() << endl;
    }
}

void EXPERIMENT::AverageReward()
{
    cout << "Main runs" << endl;
    OutputFile << "Simulations\tSteps\tAverage reward\tAverage time\n";

    SearchParams.MaxDepth = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);
    ExpParams.SimSteps = Simulator.GetHorizon(ExpParams.Accuracy, ExpParams.UndiscountedHorizon);

    for (int i = ExpParams.MinDoubles; i <= ExpParams.MaxDoubles; i++)
    {
        SearchParams.NumSimulations = 1 << i;
        SearchParams.NumStartStates = 1 << i;
        if (i + ExpParams.TransformDoubles >= 0)
            SearchParams.NumTransforms = 1 << (i + ExpParams.TransformDoubles);
        else
            SearchParams.NumTransforms = 1;
        SearchParams.MaxAttempts = SearchParams.NumTransforms * ExpParams.TransformAttempts;

        Results.Clear();
        Run();

        cout << "Simulations = " << SearchParams.NumSimulations << endl
            << "Steps = " << Results.Reward.GetCount() << endl
            << "Average reward = " << Results.Reward.GetMean()
            << " +- " << Results.Reward.GetStdErr() << endl
            << "Average time = " << Results.Time.GetMean() / Results.Reward.GetCount() << endl;
        OutputFile << SearchParams.NumSimulations << "\t"
            << Results.Reward.GetCount() << "\t"
            << Results.Reward.GetMean() << "\t"
            << Results.Reward.GetStdErr() << "\t"
            << Results.Time.GetMean() / Results.Reward.GetCount() << endl;
    }
}

//----------------------------------------------------------------------------
