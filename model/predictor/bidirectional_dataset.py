#!/usr/bin/env python3
"""
Dataset wrapper for the Bidirectional Autoregressive Transformer.

이 데이터셋은 양방향 모델에 필요한 형식으로 학습 데이터를 준비합니다:
- 초기 이미지와 상태 (샘플링된 순방향 세그먼트의 시작)
- 순방향 궤적 상태들 (샘플링된 순방향 세그먼트의 시작부터)
- 목표 이미지 (에피소드의 실제 끝에서 가져옴)
- 역방향 궤적 상태들 (에피소드의 실제 끝에서부터 시작)

This dataset prepares training data in the format required by the bidirectional model:
- Initial image and state (start of a sampled forward segment)
- Forward trajectory states (from the start of the sampled forward segment)
- Goal image (from the TRUE END of the episode)
- Backward trajectory states (starting from the TRUE END of the episode)
"""

import torch
from typing import Dict
from torch.utils.data import Dataset


class BidirectionalTrajectoryDataset(Dataset):
    """
    양방향 자기회귀 궤적 학습을 위한 데이터셋 래퍼입니다.
    이 버전은 시간적 이력(temporal history)을 지원하고 짧은 궤적에 패딩을 적용합니다.

    Dataset wrapper for bidirectional autoregressive trajectory learning.
    This version supports temporal history and pads short trajectories.
    """

    def __init__(
        self,
        lerobot_dataset,
        forward_steps: int = 16,
        backward_steps: int = 16,
        n_obs_steps: int = 1,
        image_key: str = "observation.image",
        state_key: str = "observation.state"
    ):
        """
        양방향 데이터셋을 초기화합니다.

        Args:
            lerobot_dataset: 기본 LERobot 데이터셋
            forward_steps: 순방향 궤적의 길이
            backward_steps: 역방향 궤적의 길이
            n_obs_steps: 초기 관측에 포함할 시간적 이력 단계 수
            image_key: 이미지 데이터가 저장된 키
            state_key: 상태 데이터가 저장된 키
        """
        self.lerobot_dataset = lerobot_dataset
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.n_obs_steps = n_obs_steps

        self.image_key = image_key
        self.state_key = state_key

        # Create valid trajectory samples
        self.samples = self._create_samples()

    def _create_samples(self):
        """
        데이터셋에서 유효한 학습 샘플을 생성합니다.
        Create valid training samples from the dataset.
        """
        samples = []
        episode_data_index = getattr(
            self.lerobot_dataset, 'episode_data_index', None)

        if episode_data_index and 'from' in episode_data_index and 'to' in episode_data_index:
            num_episodes = len(episode_data_index['from'])
            for episode_idx in range(num_episodes):
                from_idx = episode_data_index['from'][episode_idx].item()

                # Adjust to_idx to be inclusive (exclusive index -1 to make inclusive)
                to_idx = episode_data_index['to'][episode_idx].item() - 1

                episode_length = to_idx - from_idx + 1

                # --- 핵심 수정 부분 ---
                # 과거 이력을 확보할 수 있는 모든 지점에서 샘플링을 시작하도록 허용합니다.
                min_start_offset = self.n_obs_steps - 1
                # 에피소드의 마지막까지 시작점으로 삼을 수 있도록 max_start_offset 변경
                # 이전: episode_length - self.forward_steps → 지금: episode_length - 1
                max_start_offset = episode_length - 1

                if max_start_offset < min_start_offset:
                    # 에피소드 길이가 n_obs_steps보다 짧아 이력조차 만들 수 없는 경우만 건너뜀
                    continue

                # 에피소드 내에서 유효한 모든 시작점에 대해 샘플 생성
                # (stride는 기존과 동일하게 self.forward_steps // 2 사용)
                for start_offset in range(min_start_offset, max_start_offset + 1, self.forward_steps // 2 if self.forward_steps > 1 else 1):
                    current_start_idx = from_idx + start_offset
                    sample = {
                        'episode_idx': episode_idx,
                        'start_idx_forward_segment': current_start_idx,
                        'episode_true_end_idx': to_idx
                    }
                    samples.append(sample)
        else:
            # Fallback: This part might be less accurate if episode_index is not contiguous or well-defined
            print("Warning: `episode_data_index` not found or incomplete in `lerobot_dataset`. Falling back to `episode_index` grouping if available, which might be less robust.")
            # 경고: `episode_data_index`가 `lerobot_dataset`에서 찾을 수 없거나 불완전합니다.
            # 가능한 경우 `episode_index` 그룹화로 대체하지만, 이는 덜 안정적일 수 있습니다.

            episode_map = {}
            for i in range(len(self.lerobot_dataset)):
                try:
                    item = self.lerobot_dataset[i]  # This can be slow
                    ep_idx = item.get('episode_index', 0)
                    if ep_idx not in episode_map:
                        episode_map[ep_idx] = {'indices': []}
                    episode_map[ep_idx]['indices'].append(i)
                except Exception:
                    continue  # Skip problematic items

            for ep_idx, data in episode_map.items():
                if not data['indices']:
                    continue
                from_idx = min(data['indices'])
                to_idx = max(data['indices'])
                episode_length = to_idx - from_idx + 1

                min_start_offset = self.n_obs_steps - 1  # 시간적 이력을 위한 최소 오프셋
                # 주요 경로와 일치하도록 max_start_offset 업데이트
                max_start_offset = episode_length - 1
                if max_start_offset < min_start_offset:
                    continue

                for start_offset in range(min_start_offset, max_start_offset + 1, self.forward_steps // 2 if self.forward_steps > 1 else 1):
                    current_start_idx = from_idx + start_offset
                    sample = {
                        'episode_idx': ep_idx,  # Using the grouped episode index
                        'start_idx_forward_segment': current_start_idx,
                        'episode_true_end_idx': to_idx
                    }
                    samples.append(sample)

        print(f"Created {len(samples)} bidirectional trajectory samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        데이터셋에서 항목을 검색합니다.

        Args:
            idx: 검색할 샘플의 인덱스

        Returns:
            초기 이미지, 초기 상태, 순방향 궤적 상태, 목표 이미지, 역방향 궤적 상태가 포함된 딕셔너리
        """
        sample_info = self.samples[idx]

        # --- 에피소드 시작/끝 인덱스 정보 가져오기 ---
        episode_abs_start_idx = None
        episode_true_end_idx = sample_info['episode_true_end_idx']

        # 안전하게 에피소드 시작 인덱스 가져오기
        try:
            if hasattr(self.lerobot_dataset, 'episode_data_index') and 'from' in self.lerobot_dataset.episode_data_index:
                episode_abs_start_idx = self.lerobot_dataset.episode_data_index['from'][sample_info['episode_idx']].item(
                )
        except (AttributeError, KeyError, IndexError):
            # 에피소드 데이터 인덱스가 없는 경우 추정값 사용
            episode_abs_start_idx = max(
                0, episode_true_end_idx - self.backward_steps)

        # --- 초기 이미지와 상태(시간적 이력) ---
        # --- Initial image(s) and state(s) (temporal history) ---
        if self.n_obs_steps > 1:
            # 시간적 시퀀스: start_idx_forward_segment로 끝나는 n_obs_steps 관측값 수집
            initial_images = []
            initial_states = []

            for i in range(self.n_obs_steps):
                # start_idx_forward_segment에서 역방향으로 작업
                obs_idx = sample_info['start_idx_forward_segment'] - \
                    (self.n_obs_steps - 1 - i)

                # 경계 확인 - 에피소드 시작 전으로 이동하면 첫 번째 유효한 관측값 반복
                if episode_abs_start_idx is not None and obs_idx < episode_abs_start_idx:
                    obs_idx = episode_abs_start_idx
                elif obs_idx < 0:
                    obs_idx = 0
                elif obs_idx >= len(self.lerobot_dataset):
                    obs_idx = len(self.lerobot_dataset) - 1

                obs_data = self.lerobot_dataset[obs_idx]
                initial_images.append(obs_data[self.image_key])
                initial_states.append(obs_data[self.state_key])

            # 텐서로 변환: 이미지의 경우 [n_obs_steps, C, H, W], 상태의 경우 [n_obs_steps, state_dim]
            initial_images_tensor = torch.stack(
                [torch.as_tensor(img, dtype=torch.float32) for img in initial_images])
            initial_states_tensor = torch.stack(
                [torch.as_tensor(state, dtype=torch.float32) for state in initial_states])
        else:
            # 단일 관측: 이전 버전과의 호환성
            initial_data_idx = sample_info['start_idx_forward_segment']
            initial_data = self.lerobot_dataset[initial_data_idx]
            initial_images_tensor = torch.as_tensor(
                initial_data[self.image_key], dtype=torch.float32)
            initial_states_tensor = torch.as_tensor(
                initial_data[self.state_key], dtype=torch.float32)

        # --- 순방향 궤적 상태 (패딩 포함) ---
        # --- Forward trajectory states (with padding) ---
        forward_states = []
        last_valid_state = None
        for i in range(self.forward_steps):
            step_idx = sample_info['start_idx_forward_segment'] + i

            # 현재 스텝 인덱스가 에피소드 경계 내에 있는지 확인
            if step_idx <= sample_info['episode_true_end_idx']:
                # 경계 내에 있다면 실제 데이터를 로드
                step_data = self.lerobot_dataset[step_idx]
                state = step_data[self.state_key]
                forward_states.append(state)
                last_valid_state = state  # 마지막으로 유효했던 상태를 저장
            else:
                # 경계를 벗어났다면, 마지막 유효 상태로 패딩(padding)
                # 이 로직 덕분에 시퀀스 길이를 항상 self.forward_steps로 맞출 수 있습니다.
                if last_valid_state is None:
                    # 이 경우는 start_idx 자체가 잘못된 경우지만, 방어 코드로 초기 상태 사용
                    initial_data = self.lerobot_dataset[sample_info['start_idx_forward_segment']]
                    last_valid_state = initial_data[self.state_key]
                forward_states.append(last_valid_state)

        # --- 목표 이미지 로딩 ---
        # --- Goal image loading ---
        episode_end_data_idx = sample_info['episode_true_end_idx']
        try:
            episode_end_data = self.lerobot_dataset[episode_end_data_idx]
            goal_image = episode_end_data[self.image_key]
            true_episode_end_state = episode_end_data[self.state_key]
        except (IndexError, KeyError) as e:
            # 오류 처리 - 가장 마지막 유효한 데이터 사용
            print(
                f"Error loading episode end data at idx {episode_end_data_idx}: {e}")
            # 마지막 유효한 상태의 인덱스 찾기
            valid_idx = min(episode_end_data_idx,
                            len(self.lerobot_dataset) - 1)
            episode_end_data = self.lerobot_dataset[valid_idx]
            goal_image = episode_end_data[self.image_key]
            true_episode_end_state = episode_end_data[self.state_key]

        # --- 역방향 궤적 상태 (패딩 포함) ---
        # --- Backward trajectory states (with padding) ---
        backward_states = []
        # 에피소드 데이터 인덱스에 안전하게 접근
        episode_abs_start_idx = None
        try:
            episode_abs_start_idx = self.lerobot_dataset.episode_data_index['from'][sample_info['episode_idx']].item(
            )
        except (AttributeError, KeyError, IndexError):
            # 에피소드 데이터 인덱스가 없거나 잘못된 경우
            # 추정값으로 시작 인덱스를 설정 (주의: 정확하지 않을 수 있음)
            episode_abs_start_idx = max(
                0, episode_end_data_idx - self.backward_steps)
            print(
                f"Warning: Using estimated episode start index {episode_abs_start_idx}")

        for i in range(self.backward_steps):
            current_bwd_idx = episode_end_data_idx - i
            if current_bwd_idx >= episode_abs_start_idx and current_bwd_idx < len(self.lerobot_dataset):
                step_data = self.lerobot_dataset[current_bwd_idx]
                state = step_data[self.state_key]
                backward_states.append(state)
            else:
                # 에피소드 시작점을 넘어가거나 데이터셋 범위를 벗어나면, 가장 오래된 유효 상태로 패딩
                padding_state = backward_states[-1] if backward_states else true_episode_end_state
                backward_states.append(padding_state)

        # --- 정규화된 타임스텝 정보 추가 ---
        # 현재 샘플의 시작점이 전체 에피소드 길이 중 어느 위치에 있는지 계산 (0.0 ~ 1.0)
        current_step_in_episode = sample_info['start_idx_forward_segment'] - \
            episode_abs_start_idx
        episode_length = episode_true_end_idx - episode_abs_start_idx + 1

        # 0부터 1 사이의 값으로 정규화 (방어적 코딩)
        if episode_length > 0:
            normalized_timestep = float(
                current_step_in_episode) / float(episode_length)
            # 경계값 확인 (0.0 ~ 1.0 사이로 클램핑)
            normalized_timestep = max(0.0, min(1.0, normalized_timestep))
        else:
            normalized_timestep = 0.0

        # --- 최종 텐서 변환 및 반환 ---
        # --- Final tensor conversion and return ---
        result = {
            'initial_images': initial_images_tensor,
            'initial_states': initial_states_tensor,
            'forward_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in forward_states]),
            'goal_images': torch.as_tensor(goal_image, dtype=torch.float32),
            'backward_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in backward_states]),
            # 정규화된 타임스텝 추가
            'normalized_timestep': torch.tensor(normalized_timestep, dtype=torch.float32)
        }

        return result

    @staticmethod
    def collate_fn(batch):
        """
        배치 처리를 위한 커스텀 병합(collate) 함수입니다.
        각 배치 항목에서 동일한 키를 가진 텐서를 수집하고 스택(stack)합니다.

        Args:
            batch: 데이터셋 항목의 배치

        Returns:
            병합된 배치 딕셔너리

        Custom collate function for batching.
        """
        result = {}
        # 배치의 모든 항목이 동일한 키를 갖는지 확인
        if not batch:
            return result
        first_item_keys = batch[0].keys()

        for key in first_item_keys:
            # 현재 키에 대한 모든 항목 수집
            key_items = [item[key] for item in batch if key in item]

            if not key_items:  # 배치 항목이 일관되면 발생하지 않아야 함
                continue

            if isinstance(key_items[0], torch.Tensor):
                result[key] = torch.stack(key_items)
            else:
                # 항목이 텐서가 아닌 경우 처리 (예: 메타데이터, 여기서는 일반적이지 않음)
                # 문제가 있는 패딩으로 인한 텐서 목록인 경우 추가 조정이 필요할 수 있음
                try:
                    # 스택할 수 있는 텐서 목록인 경우 스택 시도
                    # 이는 기본 시도이며, 복잡한 구조는 더 구체적인 처리가 필요할 수 있음
                    if all(isinstance(item, torch.Tensor) for item in key_items):
                        result[key] = torch.stack(key_items)
                    else:
                        # 균일한 텐서가 아닌 경우 목록으로 저장
                        result[key] = key_items
                except Exception:
                    result[key] = key_items  # 목록으로 대체

        return result
