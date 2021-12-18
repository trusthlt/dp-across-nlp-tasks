from tqdm import tqdm, trange
import torch

import numpy as np

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from test_suit.eval_measure import official_eval_squad_test
from Transformer.dataset.utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)
from Transformer.dataset.utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad, plot_pr_curve


virtual_batch_size = False

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def official_eval_meth(examples, features, all_results, gold_data_dir, EP):
  # Compute predictions
  lr = str(EP.LEARNING_RATE).replace(".", ",")
  output_prediction_file = f"{EP.preds_dir}predictions_dp{int(EP.privacy_num)}_lr{lr}_BL{EP.tuning.training_bert}_{EP.seed}.json"
  output_nbest_file = f"{EP.preds_dir}nbest_predictions_dp{int(EP.privacy_num)}_lr{lr}_BL{EP.tuning.training_bert}_{EP.seed}.json"
  output_null_log_odds_file = f"{EP.preds_dir}null_odds_dp{int(EP.privacy_num)}_lr{lr}_BL{EP.tuning.training_bert}_{EP.seed}.json"

  write_predictions(examples, features, all_results, 10,
                  30, True, output_prediction_file,
                  output_nbest_file, output_null_log_odds_file, False,
                  True, 0.0)

  # Evaluate with the official SQuAD script
  evaluate_options = EVAL_OPTS(data_file=gold_data_dir,
                              pred_file=output_prediction_file,
                              na_prob_file=output_null_log_odds_file,
                              out_image_dir=None)
            #(OPTS, is_train, is_val, len_set)
  results = evaluate_on_squad(evaluate_options)

  return results['exact'], results['f1']

  

def train(model, train_dataloader, optimizer, device, examples, features, gold_data_dir, criterion, EP):
  
  model.train()
  train_loss_set = []
  all_results = []
  epoch_iterator = tqdm(train_dataloader, desc="Iteration")

  for step, batch in enumerate(epoch_iterator):
      #if step < global_step + 1:
          #continue
      if not virtual_batch_size:
        model.zero_grad()
      batch = tuple(t.to(device) for t in batch)
      example_indices = batch[3]

      inputs = {'input_ids':       batch[0],
                  'attention_mask':  batch[1], 
                  'token_type_ids':  batch[2],  
                  'start_positions': batch[6], 
                  'end_positions':   batch[7]}
      if not EP.use_rnn:
        outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['start_positions'], inputs['end_positions'])
        loss = outputs[0]
        train_loss_set.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      else:
        outputs, start_logits, end_logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['start_positions'], inputs['end_positions'])

        sent_batch = outputs.size()[0] * outputs.size()[1] # sent len * batch size
        out_dim = outputs.size()[2] # output dim
        outputs = torch.reshape(outputs, (sent_batch, out_dim))

        start_loss = criterion(start_logits, inputs['start_positions'])
        end_loss = criterion(end_logits, inputs['end_positions'])
        total_loss = (start_loss + end_loss) / 2
        print(total_loss)
        total_loss.backward()
        train_loss_set.append(total_loss.item())

      # batch size
      if virtual_batch_size:
        if step % 32/EP.BATCH_SIZE == 0: # change it to 31 --> same job as with 0
          optimizer.step()
          model.zero_grad()
      else:
        optimizer.step()
        
        
      # save the predictions to get an accuracy score
      for i, example_index in enumerate(example_indices):
        eval_feature = features[example_index.item()]
        unique_id = int(eval_feature.unique_id)
        if EP.use_rnn:
          result = RawResult(unique_id    = unique_id,
                            start_logits = to_list(start_logits[i]),
                            end_logits   = to_list(end_logits[i]))
        else:
          result = RawResult(unique_id    = unique_id,
                            start_logits = to_list(outputs[1][i]),
                            end_logits   = to_list(outputs[2][i]))
        all_results.append(result)

  exact, f1 = official_eval_meth(examples, features, all_results, gold_data_dir, EP)
  return exact, f1, np.mean(train_loss_set)


def evaluate(model, tokenizer, test_examples, device, validation_dataloader, features, is_test, criterion, EP, gold_data_dir=None, output_test=None):

  all_results = []
  model.eval()
  train_loss_set = []
  for batch in tqdm(validation_dataloader, desc="Evaluating", miniters=100, mininterval=5.0):
    
    batch = tuple(t.to(device) for t in batch)
    
    if not is_test:
      with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2],
                  'start_positions': batch[6], 
                  'end_positions':   batch[7]
                  }
        example_indices = batch[3]
        if EP.use_rnn:
          outputs, start_logits, end_logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['start_positions'], inputs['end_positions'])
          sent_batch = outputs.size()[0] * outputs.size()[1] # sent len * batch size
          out_dim = outputs.size()[2] # output dim
          outputs = torch.reshape(outputs, (sent_batch, out_dim))

          start_loss = criterion(start_logits, inputs['start_positions'])
          end_loss = criterion(end_logits, inputs['end_positions'])
          total_loss = (start_loss + end_loss) / 2
          train_loss_set.append(total_loss.item())
        else:
          outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['start_positions'], inputs['end_positions'])
          loss = outputs[0]
          train_loss_set.append(loss.item())
          index_start = 1
      
    else:
      with torch.no_grad():
          inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    }
          example_indices = batch[3]
          if EP.use_rnn:
            outputs, start_logits, end_logits = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
            sent_batch = outputs.size()[0] * outputs.size()[1] # sent len * batch size
            out_dim = outputs.size()[2] # output dim
            outputs = torch.reshape(outputs, (sent_batch, out_dim))
            #y = y.view(-1)

            #loss = criterion(outputs, y)
            #train_loss_set.append(loss.item())
          else:
            outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
            index_start = 0

    # save the predictions to get an accuracy score
    for i, example_index in enumerate(example_indices):
      eval_feature = features[example_index.item()]
      unique_id = int(eval_feature.unique_id)
      
      if EP.use_rnn:
          result = RawResult(unique_id    = unique_id,
                            start_logits = to_list(start_logits[i]),
                            end_logits   = to_list(end_logits[i]))
      else:
        result = RawResult(unique_id    = unique_id,
                          start_logits = to_list(outputs[index_start][i]),
                          end_logits   = to_list(outputs[index_start+1][i]))
      all_results.append(result)

  # check if the testbench of squad is calling it
  if not output_test == None:
    official_eval_squad_test(test_examples, features, all_results, EP, output_test)
    return
  else:
    exact, f1 = official_eval_meth(test_examples, features, all_results, gold_data_dir, EP)

  if not is_test:
    return exact, f1, np.mean(train_loss_set)
  else:
    return exact, f1